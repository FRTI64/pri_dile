// A from-scratch, standard-conforming ordered associative container family.
//
// Architecture:
//   * Standard Wrapper + Closure Core
//   * The red-black engine is created by a stateless generic lambda factory.
//   * The engine is a local, unnamed (Voldemort) type.
//   * All tree algorithms are generic lambdas stored in the engine tuple.
//   * The closure declares no local structs or classes: the node and engine are
//     standard tuples, and a message-dispatching lens closure is the only code
//     that touches node fields.
//   * The public containers are ordinary global class templates; the engine's
//     exact local type is obtained with decltype(factory<...>) and stored as a
//     private member of their implementation base.
//
// C++23 is required.

#include <algorithm>
#include <cassert>
#include <compare>
#include <concepts>
#include <cstddef>
#include <functional>
#include <initializer_list>
#include <iterator>
#include <memory>
#include <memory_resource>
#include <map>
#include <set>
#include <new>
#include <random>
#include <ranges>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

// ---------------------------------------------------------------------------
// Small, stateless key extractor for map-like containers.  It is deliberately
// outside the closure core; the core itself has no namespace and no globals.
// ---------------------------------------------------------------------------
struct closure_key_first
{
    template <class Pair>
    constexpr const auto &operator()(const Pair &p) const noexcept
    {
        return p.first;
    }
};

// ---------------------------------------------------------------------------
// The closure-core factory.  Every invocation produces a *new local type*.
// Five metadata parameters are the complete semantic configuration:
//   Key, Value, KeyOfValue, Compare, AllowDuplicates.
// The allocator is intentionally not part of the engine: the wrapper owns the
// allocator and supplies allocator-aware node lifetime operations.  This keeps
// the engine stateless with respect to memory resources and lets node_type
// carry the exact allocator required by the standard.
//
// The core is built exclusively from closures and standard-library aggregates;
// it declares no local struct or class.  A node is a tuple whose link fields
// are void* so that the tuple type does not need to name itself.  The engine is
// a tuple of state and generic-lambda operations, addressed by index.
// ---------------------------------------------------------------------------
inline constexpr auto make_stl_toolkit =
    []<typename Key, typename Value, typename KeyOfValue, typename Compare,
       bool AllowDuplicates>(Compare comp = Compare{})
{
    // Node layout: parent, left, right, next, prev, red, engaged, value.
    // The value is raw, correctly aligned storage; the wrapper constructs and
    // destroys it through the value allocator.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
    using value_storage =
        std::aligned_storage_t<sizeof(Value), alignof(Value)>;
#pragma GCC diagnostic pop
    using node =
        std::tuple<void *, void *, void *, void *, void *, bool, bool,
                   value_storage>;

    constexpr std::size_t n_parent = 0;
    constexpr std::size_t n_left = 1;
    constexpr std::size_t n_right = 2;
    constexpr std::size_t n_next = 3;
    constexpr std::size_t n_prev = 4;
    constexpr std::size_t n_red = 5;
    constexpr std::size_t n_engaged = 6;
    constexpr std::size_t n_value = 7;

    // Core layout: header, count, comparator, key extractor, lens function,
    // successor function, predecessor function, then every algorithm closure.
    constexpr std::size_t c_header = 0;
    constexpr std::size_t c_count = 1;
    constexpr std::size_t c_comp = 2;
    constexpr std::size_t c_key_of = 3;

    // Message tags.  Aliases of std::integral_constant keep the dispatcher
    // entirely closure-based without declaring local tag types.
    using left_tag = std::integral_constant<int, 0>;
    using right_tag = std::integral_constant<int, 1>;
    using parent_tag = std::integral_constant<int, 2>;
    using red_tag = std::integral_constant<int, 3>;
    using engaged_tag = std::integral_constant<int, 4>;
    using value_tag = std::integral_constant<int, 5>;
    using next_tag = std::integral_constant<int, 6>;
    using prev_tag = std::integral_constant<int, 7>;

    using set_left_tag = std::integral_constant<int, 8>;
    using set_right_tag = std::integral_constant<int, 9>;
    using set_parent_tag = std::integral_constant<int, 10>;
    using set_red_tag = std::integral_constant<int, 11>;
    using set_engaged_tag = std::integral_constant<int, 12>;
    using set_next_tag = std::integral_constant<int, 13>;
    using set_prev_tag = std::integral_constant<int, 14>;

    // -------------------------------------------------------------------
    // Functional lens.  This is the only code in the entire program which
    // reads or writes node fields.  Algorithms ask for a message tag; they
    // never see a field.  wrap(node) is a higher-order closure: it closes
    // over only the node and returns another generic closure.
    // -------------------------------------------------------------------
    static constinit auto wrap_fn = [](node *p) noexcept
    {
        return [p]<class Tag, class... Args>(Tag,
                                             Args &&...args) noexcept
                   -> decltype(auto)
        {
            using T = std::remove_cvref_t<Tag>;
            if constexpr (std::same_as<T, left_tag>)
            {
                return static_cast<node *>(std::get<n_left>(*p));
            }
            else if constexpr (std::same_as<T, right_tag>)
            {
                return static_cast<node *>(std::get<n_right>(*p));
            }
            else if constexpr (std::same_as<T, parent_tag>)
            {
                return static_cast<node *>(std::get<n_parent>(*p));
            }
            else if constexpr (std::same_as<T, red_tag>)
            {
                return static_cast<bool>(std::get<n_red>(*p));
            }
            else if constexpr (std::same_as<T, engaged_tag>)
            {
                return static_cast<bool>(std::get<n_engaged>(*p));
            }
            else if constexpr (std::same_as<T, value_tag>)
            {
                return (*std::launder(reinterpret_cast<Value *>(
                    std::addressof(std::get<n_value>(*p)))));
            }
            else if constexpr (std::same_as<T, next_tag>)
            {
                return static_cast<node *>(std::get<n_next>(*p));
            }
            else if constexpr (std::same_as<T, prev_tag>)
            {
                return static_cast<node *>(std::get<n_prev>(*p));
            }
            else if constexpr (std::same_as<T, set_left_tag>)
            {
                static_assert(sizeof...(Args) == 1);
                std::get<n_left>(*p) =
                    static_cast<void *>((std::forward<Args>(args), ...));
            }
            else if constexpr (std::same_as<T, set_right_tag>)
            {
                static_assert(sizeof...(Args) == 1);
                std::get<n_right>(*p) =
                    static_cast<void *>((std::forward<Args>(args), ...));
            }
            else if constexpr (std::same_as<T, set_parent_tag>)
            {
                static_assert(sizeof...(Args) == 1);
                std::get<n_parent>(*p) =
                    static_cast<void *>((std::forward<Args>(args), ...));
            }
            else if constexpr (std::same_as<T, set_red_tag>)
            {
                static_assert(sizeof...(Args) == 1);
                std::get<n_red>(*p) = (std::forward<Args>(args), ...);
            }
            else if constexpr (std::same_as<T, set_engaged_tag>)
            {
                static_assert(sizeof...(Args) == 1);
                std::get<n_engaged>(*p) = (std::forward<Args>(args), ...);
            }
            else if constexpr (std::same_as<T, set_next_tag>)
            {
                static_assert(sizeof...(Args) == 1);
                std::get<n_next>(*p) =
                    static_cast<void *>((std::forward<Args>(args), ...));
            }
            else if constexpr (std::same_as<T, set_prev_tag>)
            {
                static_assert(sizeof...(Args) == 1);
                std::get<n_prev>(*p) =
                    static_cast<void *>((std::forward<Args>(args), ...));
            }
            else
            {
                static_assert(!sizeof(Tag), "unknown node-lens message");
            }
        };
    };

    // -------------------------------------------------------------------
    // Iteration is physically O(1): every node stores its in-order
    // predecessor and successor.  These two lambdas expose those links via
    // the lens and are stored in the engine as ordinary function pointers,
    // so iterator objects survive container moves and swaps.
    // -------------------------------------------------------------------
    static constinit auto next_lambda_fn = [](node *p) noexcept -> node *
    {
        return wrap_fn(p)(next_tag{});
    };
    static constinit auto prev_lambda_fn = [](node *p) noexcept -> node *
    {
        return wrap_fn(p)(prev_tag{});
    };

    // -------------------------------------------------------------------
    // Rotations (pure closures).
    // -------------------------------------------------------------------
    static constinit auto left_rotate_fn = [](auto &self,
                                               node *x) noexcept
    {
        auto L = wrap_fn;
        auto &header = std::get<c_header>(self);
        node *y = L(x)(right_tag{});
        L(x)(set_right_tag{}, L(y)(left_tag{}));
        if (L(y)(left_tag{}))
        {
            L(L(y)(left_tag{}))(set_parent_tag{}, x);
        }
        node *xp = L(x)(parent_tag{});
        L(y)(set_parent_tag{}, xp);
        if (xp == &header)
        {
            L (&header)(set_left_tag{}, y);
        }
        else if (x == L(xp)(left_tag{}))
        {
            L(xp)(set_left_tag{}, y);
        }
        else
        {
            L(xp)(set_right_tag{}, y);
        }
        L(y)(set_left_tag{}, x);
        L(x)(set_parent_tag{}, y);
    };

    static constinit auto right_rotate_fn = [](auto &self,
                                                node *x) noexcept
    {
        auto L = wrap_fn;
        auto &header = std::get<c_header>(self);
        node *y = L(x)(left_tag{});
        L(x)(set_left_tag{}, L(y)(right_tag{}));
        if (L(y)(right_tag{}))
        {
            L(L(y)(right_tag{}))(set_parent_tag{}, x);
        }
        node *xp = L(x)(parent_tag{});
        L(y)(set_parent_tag{}, xp);
        if (xp == &header)
        {
            L (&header)(set_left_tag{}, y);
        }
        else if (x == L(xp)(right_tag{}))
        {
            L(xp)(set_right_tag{}, y);
        }
        else
        {
            L(xp)(set_left_tag{}, y);
        }
        L(y)(set_right_tag{}, x);
        L(x)(set_parent_tag{}, y);
    };

    static constinit auto minimum_fn = [](auto &,
                                           node *x) noexcept -> node *
    {
        auto L = wrap_fn;
        while (x && L(x)(left_tag{}))
        {
            x = L(x)(left_tag{});
        }
        return x;
    };

    static constinit auto maximum_fn = [](auto &,
                                           node *x) noexcept -> node *
    {
        auto L = wrap_fn;
        while (x && L(x)(right_tag{}))
        {
            x = L(x)(right_tag{});
        }
        return x;
    };

    // -------------------------------------------------------------------
    // Insertion fix-up (CLRS), expressed as a closure.
    // -------------------------------------------------------------------
    static constinit auto insert_fix_fn = [](auto &self, node *z) noexcept
    {
        auto L = wrap_fn;
        auto &header = std::get<c_header>(self);
        auto root = [&]() noexcept
        {
            return L(&header)(left_tag{});
        };
        auto is_red = [&](node *n) noexcept
        {
            return n != nullptr && L(n)(red_tag{});
        };

        while (z != root() && is_red(L(z)(parent_tag{})))
        {
            node *p = L(z)(parent_tag{});
            node *g = L(p)(parent_tag{});
            if (p == L(g)(left_tag{}))
            {
                node *u = L(g)(right_tag{});
                if (is_red(u))
                {
                    L(p)(set_red_tag{}, false);
                    L(u)(set_red_tag{}, false);
                    L(g)(set_red_tag{}, true);
                    z = g;
                }
                else
                {
                    if (z == L(p)(right_tag{}))
                    {
                        z = p;
                        left_rotate_fn(self, z);
                        p = L(z)(parent_tag{});
                        g = L(p)(parent_tag{});
                    }
                    L(p)(set_red_tag{}, false);
                    L(g)(set_red_tag{}, true);
                    right_rotate_fn(self, g);
                }
            }
            else
            {
                node *u = L(g)(left_tag{});
                if (is_red(u))
                {
                    L(p)(set_red_tag{}, false);
                    L(u)(set_red_tag{}, false);
                    L(g)(set_red_tag{}, true);
                    z = g;
                }
                else
                {
                    if (z == L(p)(left_tag{}))
                    {
                        z = p;
                        right_rotate_fn(self, z);
                        p = L(z)(parent_tag{});
                        g = L(p)(parent_tag{});
                    }
                    L(p)(set_red_tag{}, false);
                    L(g)(set_red_tag{}, true);
                    left_rotate_fn(self, g);
                }
            }
        }
        if (root())
        {
            L(root())(set_red_tag{}, false);
        }
    };

    // -------------------------------------------------------------------
    // Deletion fix-up.  x may be null, in which case xp is its parent.
    // -------------------------------------------------------------------
    static constinit auto erase_fix_fn = [](auto &self, node *x,
                                             node *xp) noexcept
    {
        auto L = wrap_fn;
        auto &header = std::get<c_header>(self);
        auto is_red = [&](node *n) noexcept
        {
            return n != nullptr && L(n)(red_tag{});
        };
        while (x != L(&header)(left_tag{}) && !is_red(x))
        {
            if (xp == nullptr)
            {
                break;
            }
            if (x == L(xp)(left_tag{}))
            {
                node *w = L(xp)(right_tag{});
                if (w == nullptr)
                {
                    x = xp;
                    xp = L(x)(parent_tag{});
                    continue;
                }
                if (is_red(w))
                {
                    L(w)(set_red_tag{}, false);
                    L(xp)(set_red_tag{}, true);
                    left_rotate_fn(self, xp);
                    w = L(xp)(right_tag{});
                }
                if (!is_red(L(w)(left_tag{})) &&
                    !is_red(L(w)(right_tag{})))
                {
                    L(w)(set_red_tag{}, true);
                    x = xp;
                    xp = L(x)(parent_tag{});
                }
                else
                {
                    if (!is_red(L(w)(right_tag{})))
                    {
                        if (L(w)(left_tag{}))
                        {
                            L(L(w)(left_tag{}))(set_red_tag{}, false);
                        }
                        L(w)(set_red_tag{}, true);
                        right_rotate_fn(self, w);
                        w = L(xp)(right_tag{});
                    }
                    L(w)(set_red_tag{}, L(xp)(red_tag{}));
                    L(xp)(set_red_tag{}, false);
                    if (L(w)(right_tag{}))
                    {
                        L(L(w)(right_tag{}))(set_red_tag{}, false);
                    }
                    left_rotate_fn(self, xp);
                    x = L(&header)(left_tag{});
                    xp = nullptr;
                }
            }
            else
            {
                node *w = L(xp)(left_tag{});
                if (w == nullptr)
                {
                    x = xp;
                    xp = L(x)(parent_tag{});
                    continue;
                }
                if (is_red(w))
                {
                    L(w)(set_red_tag{}, false);
                    L(xp)(set_red_tag{}, true);
                    right_rotate_fn(self, xp);
                    w = L(xp)(left_tag{});
                }
                if (!is_red(L(w)(right_tag{})) &&
                    !is_red(L(w)(left_tag{})))
                {
                    L(w)(set_red_tag{}, true);
                    x = xp;
                    xp = L(x)(parent_tag{});
                }
                else
                {
                    if (!is_red(L(w)(left_tag{})))
                    {
                        if (L(w)(right_tag{}))
                        {
                            L(L(w)(right_tag{}))(set_red_tag{}, false);
                        }
                        L(w)(set_red_tag{}, true);
                        left_rotate_fn(self, w);
                        w = L(xp)(left_tag{});
                    }
                    L(w)(set_red_tag{}, L(xp)(red_tag{}));
                    L(xp)(set_red_tag{}, false);
                    if (L(w)(left_tag{}))
                    {
                        L(L(w)(left_tag{}))(set_red_tag{}, false);
                    }
                    right_rotate_fn(self, xp);
                    x = L(&header)(left_tag{});
                    xp = nullptr;
                }
            }
        }
        if (x)
        {
            L(x)(set_red_tag{}, false);
        }
    };

    static constinit auto transplant_fn = [](auto &self, node *u,
                                              node *v) noexcept
    {
        auto L = wrap_fn;
        auto &header = std::get<c_header>(self);
        node *up = L(u)(parent_tag{});
        if (up == &header)
        {
            L (&header)(set_left_tag{}, v);
        }
        else if (u == L(up)(left_tag{}))
        {
            L(up)(set_left_tag{}, v);
        }
        else
        {
            L(up)(set_right_tag{}, v);
        }
        if (v)
        {
            L(v)(set_parent_tag{}, up);
        }
    };

    // -------------------------------------------------------------------
    // Insertion.  Equivalent keys are linked after the existing equivalent
    // range, preserving stable order for multi containers.
    // -------------------------------------------------------------------
    static constinit auto insert_node_fn = [](auto &self, node *z)
        -> std::pair<node *, bool>
    {
        auto L = wrap_fn;
        auto &header = std::get<c_header>(self);
        auto &comp = std::get<c_comp>(self);
        auto &key_of = std::get<c_key_of>(self);
        auto &count = std::get<c_count>(self);
        node *y = &header;
        node *x = L(&header)(left_tag{});
        bool went_left = false;

        while (x)
        {
            y = x;
            const Key &zk = key_of(L(z)(value_tag{}));
            const Key &xk = key_of(L(x)(value_tag{}));
            const bool before = comp(zk, xk);
            if constexpr (!AllowDuplicates)
            {
                const bool after = comp(xk, zk);
                if (!before && !after)
                {
                    return {x, false};
                }
            }
            if (before)
            {
                x = L(x)(left_tag{});
                went_left = true;
            }
            else
            {
                x = L(x)(right_tag{});
                went_left = false;
            }
        }

        L(z)(set_parent_tag{}, y);
        L(z)(set_left_tag{}, static_cast<node *>(nullptr));
        L(z)(set_right_tag{}, static_cast<node *>(nullptr));
        L(z)(set_red_tag{}, true);

        if (y == &header)
        {
            L (&header)(set_left_tag{}, z);
            L (&header)(set_parent_tag{}, z);
            L(z)(set_next_tag{}, &header);
            L(z)(set_prev_tag{}, &header);
            L (&header)(set_next_tag{}, z);
            L (&header)(set_prev_tag{}, z);
        }
        else if (went_left)
        {
            L(y)(set_left_tag{}, z);
            node *pv = L(y)(prev_tag{});
            L(z)(set_next_tag{}, y);
            L(z)(set_prev_tag{}, pv);
            L(pv)(set_next_tag{}, z);
            L(y)(set_prev_tag{}, z);
        }
        else
        {
            L(y)(set_right_tag{}, z);
            node *nx = L(y)(next_tag{});
            L(z)(set_prev_tag{}, y);
            L(z)(set_next_tag{}, nx);
            L(nx)(set_prev_tag{}, z);
            L(y)(set_next_tag{}, z);
        }

        ++count;
        insert_fix_fn(self, z);
        node *r = L(&header)(left_tag{});
        L (&header)(set_parent_tag{}, r);
        return {z, true};
    };

    // Hinted insertion: use the hint when it really describes a valid
    // neighbouring position (amortized constant), otherwise fall back to
    // the ordinary logarithmic search.
    static constinit auto insert_node_hint_fn = [](auto &self, node *z,
                                                    node *hint)
        -> std::pair<node *, bool>
    {
        auto L = wrap_fn;
        auto &header = std::get<c_header>(self);
        auto &comp = std::get<c_comp>(self);
        auto &key_of = std::get<c_key_of>(self);
        auto &count = std::get<c_count>(self);
        node *root = L(&header)(left_tag{});
        if (!root)
        {
            return insert_node_fn(self, z);
        }
        const Key &zk = key_of(L(z)(value_tag{}));

        if (hint == &header)
        {
            node *last = L(&header)(prev_tag{});
            const Key &lk = key_of(L(last)(value_tag{}));
            const bool valid =
                AllowDuplicates ? !comp(zk, lk) : comp(lk, zk);
            if (!valid)
            {
                return insert_node_fn(self, z);
            }
            L(z)(set_parent_tag{}, last);
            L(z)(set_left_tag{}, static_cast<node *>(nullptr));
            L(z)(set_right_tag{}, static_cast<node *>(nullptr));
            L(z)(set_red_tag{}, true);
            L(last)(set_right_tag{}, z);
            L(z)(set_prev_tag{}, last);
            L(z)(set_next_tag{}, &header);
            L(last)(set_next_tag{}, z);
            L (&header)(set_prev_tag{}, z);
            ++count;
            insert_fix_fn(self, z);
            L (&header)(set_parent_tag{},
                        L (&header)(left_tag{}));
            return {z, true};
        }

        const Key &hk = key_of(L(hint)(value_tag{}));
        if constexpr (AllowDuplicates)
        {
            if (comp(hk, zk))
            {
                return insert_node_fn(self, z);
            }
        }
        else
        {
            if (!comp(zk, hk))
            {
                return insert_node_fn(self, z);
            }
        }

        node *pred = L(hint)(prev_tag{});
        if (pred != &header)
        {
            const Key &pk = key_of(L(pred)(value_tag{}));
            if constexpr (AllowDuplicates)
            {
                if (comp(zk, pk))
                {
                    return insert_node_fn(self, z);
                }
            }
            else
            {
                if (!comp(pk, zk))
                {
                    return insert_node_fn(self, z);
                }
            }
        }

        node *parent = nullptr;
        if (L(hint)(left_tag{}) == nullptr)
        {
            parent = hint;
            L(hint)(set_left_tag{}, z);
        }
        else
        {
            parent = maximum_fn(self, L(hint)(left_tag{}));
            L(parent)(set_right_tag{}, z);
        }
        L(z)(set_parent_tag{}, parent);
        L(z)(set_left_tag{}, static_cast<node *>(nullptr));
        L(z)(set_right_tag{}, static_cast<node *>(nullptr));
        L(z)(set_red_tag{}, true);
        L(z)(set_prev_tag{}, pred);
        L(z)(set_next_tag{}, hint);
        L(pred)(set_next_tag{}, z);
        L(hint)(set_prev_tag{}, z);
        ++count;
        insert_fix_fn(self, z);
        L (&header)(set_parent_tag{}, L (&header)(left_tag{}));
        return {z, true};
    };

    // -------------------------------------------------------------------
    // Deletion / extraction.  Thread links are repaired in O(1).
    // -------------------------------------------------------------------
    static constinit auto detach_fn = [](auto &self, node *z) noexcept
    {
        auto L = wrap_fn;
        auto &header = std::get<c_header>(self);
        auto &count = std::get<c_count>(self);
        node *zn = L(z)(next_tag{});
        node *zp = L(z)(prev_tag{});
        L(zp)(set_next_tag{}, zn);
        L(zn)(set_prev_tag{}, zp);

        node *y = z;
        bool y_was_red = L(y)(red_tag{});
        node *x = nullptr;
        node *xp = nullptr;

        if (L(z)(left_tag{}) == nullptr)
        {
            x = L(z)(right_tag{});
            xp = L(z)(parent_tag{});
            transplant_fn(self, z, x);
        }
        else if (L(z)(right_tag{}) == nullptr)
        {
            x = L(z)(left_tag{});
            xp = L(z)(parent_tag{});
            transplant_fn(self, z, x);
        }
        else
        {
            y = minimum_fn(self, L(z)(right_tag{}));
            y_was_red = L(y)(red_tag{});
            x = L(y)(right_tag{});
            if (L(y)(parent_tag{}) == z)
            {
                xp = y;
                if (x)
                {
                    L(x)(set_parent_tag{}, y);
                }
            }
            else
            {
                xp = L(y)(parent_tag{});
                transplant_fn(self, y, x);
                L(y)(set_right_tag{}, L(z)(right_tag{}));
                L(L(y)(right_tag{}))(set_parent_tag{}, y);
            }
            transplant_fn(self, z, y);
            L(y)(set_left_tag{}, L(z)(left_tag{}));
            L(L(y)(left_tag{}))(set_parent_tag{}, y);
            L(y)(set_red_tag{}, L(z)(red_tag{}));
        }

        --count;
        if (!y_was_red)
        {
            erase_fix_fn(self, x, xp);
        }

        node *r = L(&header)(left_tag{});
        if (r)
        {
            L(r)(set_parent_tag{}, &header);
        }
        L (&header)(set_parent_tag{}, r);

        // Make z a standalone node once again.
        L(z)(set_left_tag{}, static_cast<node *>(nullptr));
        L(z)(set_right_tag{}, static_cast<node *>(nullptr));
        L(z)(set_parent_tag{}, static_cast<node *>(nullptr));
        L(z)(set_next_tag{}, static_cast<node *>(nullptr));
        L(z)(set_prev_tag{}, static_cast<node *>(nullptr));
        L(z)(set_red_tag{}, false);
    };

    // -------------------------------------------------------------------
    // Lookup closures are templates through generic lambdas, enabling
    // heterogeneous lookup when Compare::is_transparent exists.
    // -------------------------------------------------------------------
    static constinit auto lower_bound_fn = [](auto &self,
                                               const auto &key) -> node *
    {
        auto L = wrap_fn;
        auto &header = std::get<c_header>(self);
        auto &comp = std::get<c_comp>(self);
        auto &key_of = std::get<c_key_of>(self);
        node *x = L(&header)(left_tag{});
        node *y = &header;
        while (x)
        {
            const Key &xk = key_of(L(x)(value_tag{}));
            if (!comp(xk, key))
            {
                y = x;
                x = L(x)(left_tag{});
            }
            else
            {
                x = L(x)(right_tag{});
            }
        }
        return y;
    };

    static constinit auto upper_bound_fn = [](auto &self,
                                               const auto &key) -> node *
    {
        auto L = wrap_fn;
        auto &header = std::get<c_header>(self);
        auto &comp = std::get<c_comp>(self);
        auto &key_of = std::get<c_key_of>(self);
        node *x = L(&header)(left_tag{});
        node *y = &header;
        while (x)
        {
            const Key &xk = key_of(L(x)(value_tag{}));
            if (comp(key, xk))
            {
                y = x;
                x = L(x)(left_tag{});
            }
            else
            {
                x = L(x)(right_tag{});
            }
        }
        return y;
    };

    static constinit auto find_node_fn = [](auto &self,
                                             const auto &key) -> node *
    {
        auto &header = std::get<c_header>(self);
        auto &comp = std::get<c_comp>(self);
        auto &key_of = std::get<c_key_of>(self);
        node *n = lower_bound_fn(self, key);
        if (n != &header)
        {
            const Key &nk = key_of(wrap_fn(n)(value_tag{}));
            if (!comp(key, nk) && !comp(nk, key))
            {
                return n;
            }
        }
        return &header;
    };

    // -------------------------------------------------------------------
    // Structural utilities used by the wrapper (copying, clearing,
    // stealing, swapping).  They also use only the lens.
    // -------------------------------------------------------------------
    static constinit auto rebuild_threads_fn = [](auto &self) noexcept
    {
        auto L = wrap_fn;
        auto &header = std::get<c_header>(self);
        node *r = L(&header)(left_tag{});
        node *prev = &header;
        auto visit = [&](auto &&go, node *n) -> void
        {
            if (!n)
            {
                return;
            }
            go(go, L(n)(left_tag{}));
            L(prev)(set_next_tag{}, n);
            L(n)(set_prev_tag{}, prev);
            prev = n;
            go(go, L(n)(right_tag{}));
        };
        if (r)
        {
            visit(visit, r);
        }
        L(prev)(set_next_tag{}, &header);
        L (&header)(set_prev_tag{}, prev);
        L (&header)(set_next_tag{},
                    r ? minimum_fn(self, r) : &header);
        L (&header)(set_parent_tag{}, r);
    };

    static constinit auto clear_tree_fn = [](auto &, node *n,
                                              auto &&destroy) -> void
    {
        // A small Y-combinator through the first argument keeps the closure
        // recursive without naming its own (not-yet-deduced) type.
        auto visit = [](auto &&go, node *cur, auto &&d) -> void
        {
            if (!cur)
            {
                return;
            }
            auto L = wrap_fn;
            node *l = L(cur)(left_tag{});
            node *r = L(cur)(right_tag{});
            go(go, l, d);
            go(go, r, d);
            d(cur);
        };
        visit(visit, n, destroy);
    };

    static constinit auto clone_tree_fn = [](auto &self, node *src,
                                              auto &&create,
                                              auto &&destroy) -> node *
    {
        auto visit = [&](auto &&go, node *s, auto &&make,
                         auto &&drop) -> node *
        {
            if (!s)
            {
                return nullptr;
            }
            auto L = wrap_fn;
            node *d = make(s);
            try
            {
                node *l = go(go, L(s)(left_tag{}), make, drop);
                L(d)(set_left_tag{}, l);
                if (l)
                {
                    L(l)(set_parent_tag{}, d);
                }
                node *r = go(go, L(s)(right_tag{}), make, drop);
                L(d)(set_right_tag{}, r);
                if (r)
                {
                    L(r)(set_parent_tag{}, d);
                }
                L(d)(set_red_tag{}, L(s)(red_tag{}));
                return d;
            }
            catch (...)
            {
                clear_tree_fn(self, d, drop);
                throw;
            }
        };
        return visit(visit, src, create, destroy);
    };

    static constinit auto reset_fn = [](auto &self) noexcept
    {
        auto L = wrap_fn;
        auto &header = std::get<c_header>(self);
        L (&header)(set_left_tag{}, static_cast<node *>(nullptr));
        L (&header)(set_right_tag{}, static_cast<node *>(nullptr));
        L (&header)(set_parent_tag{}, static_cast<node *>(nullptr));
        L (&header)(set_next_tag{}, &header);
        L (&header)(set_prev_tag{}, &header);
        std::get<c_count>(self) = 0;
    };

    static constinit auto adopt_fn = [](auto &self, node *r, std::size_t n)
    {
        auto L = wrap_fn;
        auto &header = std::get<c_header>(self);
        L (&header)(set_left_tag{}, r);
        if (r)
        {
            L(r)(set_parent_tag{}, &header);
        }
        L (&header)(set_parent_tag{}, r);
        std::get<c_count>(self) = n;
        rebuild_threads_fn(self);
    };

    static constinit auto take_from_fn = [](auto &dst, auto &src) noexcept
    {
        auto L = wrap_fn;
        auto &dst_header = std::get<c_header>(dst);
        auto &src_header = std::get<c_header>(src);
        node *r = L(&src_header)(left_tag{});
        node *first = L(&src_header)(next_tag{});
        node *last = L(&src_header)(prev_tag{});
        L (&dst_header)(set_left_tag{}, r);
        if (r)
        {
            L(r)(set_parent_tag{}, &dst_header);
            L(first)(set_prev_tag{}, &dst_header);
            L(last)(set_next_tag{}, &dst_header);
            L (&dst_header)(set_next_tag{}, first);
            L (&dst_header)(set_prev_tag{}, last);
        }
        else
        {
            L (&dst_header)(set_next_tag{}, &dst_header);
            L (&dst_header)(set_prev_tag{}, &dst_header);
        }
        L (&dst_header)(set_parent_tag{}, r);
        std::get<c_count>(dst) = std::get<c_count>(src);
        reset_fn(src);
    };

    static constinit auto swap_trees_fn = [](auto &a, auto &b) noexcept
    {
        auto L = wrap_fn;
        auto &ah = std::get<c_header>(a);
        auto &bh = std::get<c_header>(b);
        node *ra = L(&ah)(left_tag{});
        node *rb = L(&bh)(left_tag{});
        node *af = L(&ah)(next_tag{});
        node *al = L(&ah)(prev_tag{});
        node *bf = L(&bh)(next_tag{});
        node *bl = L(&bh)(prev_tag{});

        L (&ah)(set_left_tag{}, rb);
        L (&ah)(set_parent_tag{}, rb);
        L (&ah)(set_next_tag{}, bf == &bh ? &ah : bf);
        L (&ah)(set_prev_tag{}, bl == &bh ? &ah : bl);

        L (&bh)(set_left_tag{}, ra);
        L (&bh)(set_parent_tag{}, ra);
        L (&bh)(set_next_tag{}, af == &ah ? &bh : af);
        L (&bh)(set_prev_tag{}, al == &ah ? &bh : al);

        if (ra)
        {
            L(ra)(set_parent_tag{}, &bh);
            L(af)(set_prev_tag{}, &bh);
            L(al)(set_next_tag{}, &bh);
        }
        if (rb)
        {
            L(rb)(set_parent_tag{}, &ah);
            L(bf)(set_prev_tag{}, &ah);
            L(bl)(set_next_tag{}, &ah);
        }
        std::swap(std::get<c_count>(a), std::get<c_count>(b));
    };

    static constinit auto first_node_fn = [](auto &self) noexcept
    {
        auto &header = std::get<c_header>(self);
        return wrap_fn(&header)(next_tag{});
    };
    static constinit auto last_node_fn = [](auto &self) noexcept
    {
        auto &header = std::get<c_header>(self);
        return wrap_fn(&header)(prev_tag{});
    };
    static constinit auto root_node_fn = [](auto &self) noexcept
    {
        auto &header = std::get<c_header>(self);
        return wrap_fn(&header)(left_tag{});
    };
    static constinit auto end_node_fn = [](auto &self) noexcept
    {
        return &std::get<c_header>(self);
    };

    using core = std::tuple<
        node, std::size_t, Compare, KeyOfValue,
        decltype(+wrap_fn), decltype(+next_lambda_fn),
        decltype(+prev_lambda_fn), decltype(left_rotate_fn),
        decltype(right_rotate_fn), decltype(minimum_fn),
        decltype(maximum_fn), decltype(insert_fix_fn),
        decltype(erase_fix_fn), decltype(transplant_fn),
        decltype(insert_node_fn), decltype(insert_node_hint_fn),
        decltype(detach_fn), decltype(lower_bound_fn),
        decltype(upper_bound_fn), decltype(find_node_fn),
        decltype(rebuild_threads_fn), decltype(clear_tree_fn),
        decltype(clone_tree_fn), decltype(reset_fn), decltype(adopt_fn),
        decltype(take_from_fn), decltype(swap_trees_fn),
        decltype(first_node_fn), decltype(last_node_fn),
        decltype(root_node_fn), decltype(end_node_fn)>;

    return core{
        node{},
        0,
        comp,
        KeyOfValue{},
        +wrap_fn,
        +next_lambda_fn,
        +prev_lambda_fn,
        left_rotate_fn,
        right_rotate_fn,
        minimum_fn,
        maximum_fn,
        insert_fix_fn,
        erase_fix_fn,
        transplant_fn,
        insert_node_fn,
        insert_node_hint_fn,
        detach_fn,
        lower_bound_fn,
        upper_bound_fn,
        find_node_fn,
        rebuild_threads_fn,
        clear_tree_fn,
        clone_tree_fn,
        reset_fn,
        adopt_fn,
        take_from_fn,
        swap_trees_fn,
        first_node_fn,
        last_node_fn,
        root_node_fn,
        end_node_fn,
    };
};

// ---------------------------------------------------------------------------
// Public standard wrapper.  The engine's exact local type is obtained from
// decltype and stored as a private member.  Only the wrapper knows about
// allocators; the engine knows only about five semantic parameters.
// ---------------------------------------------------------------------------
template <class Key, class Value, class KeyOfValue, class Compare,
          class Allocator, bool AllowDuplicates, bool Mapped>
class basic_ordered_tree
{
    using core_t = decltype(make_stl_toolkit.template operator()<
                            Key, Value, KeyOfValue, Compare, AllowDuplicates>(
        std::declval<Compare>()));

    // Core tuple layout.  The engine is an aggregate of closures rather than
    // a local class, so the wrapper addresses its fields by index.
    static constexpr std::size_t core_header_i = 0;
    static constexpr std::size_t core_count_i = 1;
    static constexpr std::size_t core_comp_i = 2;
    static constexpr std::size_t core_key_of_i = 3;
    static constexpr std::size_t core_lens_i = 4;
    static constexpr std::size_t core_next_i = 5;
    static constexpr std::size_t core_prev_i = 6;
    static constexpr std::size_t core_left_rotate_i = 7;
    static constexpr std::size_t core_right_rotate_i = 8;
    static constexpr std::size_t core_minimum_i = 9;
    static constexpr std::size_t core_maximum_i = 10;
    static constexpr std::size_t core_insert_fix_i = 11;
    static constexpr std::size_t core_erase_fix_i = 12;
    static constexpr std::size_t core_transplant_i = 13;
    static constexpr std::size_t core_insert_node_i = 14;
    static constexpr std::size_t core_insert_node_hint_i = 15;
    static constexpr std::size_t core_detach_i = 16;
    static constexpr std::size_t core_lower_bound_i = 17;
    static constexpr std::size_t core_upper_bound_i = 18;
    static constexpr std::size_t core_find_node_i = 19;
    static constexpr std::size_t core_rebuild_threads_i = 20;
    static constexpr std::size_t core_clear_tree_i = 21;
    static constexpr std::size_t core_clone_tree_i = 22;
    static constexpr std::size_t core_reset_i = 23;
    static constexpr std::size_t core_adopt_i = 24;
    static constexpr std::size_t core_take_from_i = 25;
    static constexpr std::size_t core_swap_trees_i = 26;
    static constexpr std::size_t core_first_node_i = 27;
    static constexpr std::size_t core_last_node_i = 28;
    static constexpr std::size_t core_root_node_i = 29;
    static constexpr std::size_t core_end_node_i = 30;

    // Node tuple layout.
    static constexpr std::size_t node_engaged_i = 6;
    static constexpr std::size_t node_value_i = 7;

    using node_t = std::tuple_element_t<core_header_i, core_t>;
    using next_fn = std::tuple_element_t<core_next_i, core_t>;
    using prev_fn = std::tuple_element_t<core_prev_i, core_t>;
    using node_allocator_type =
        typename std::allocator_traits<Allocator>::template rebind_alloc<
            node_t>;
    using node_alloc_traits = std::allocator_traits<node_allocator_type>;
    using value_allocator_type =
        typename std::allocator_traits<Allocator>::template rebind_alloc<
            Value>;
    using value_alloc_traits = std::allocator_traits<value_allocator_type>;

    static Value *value_ptr(node_t *p) noexcept
    {
        return std::launder(reinterpret_cast<Value *>(
            std::addressof(std::get<node_value_i>(*p))));
    }

    static bool engaged(node_t *p) noexcept
    {
        return std::get<node_engaged_i>(*p);
    }

    static void set_engaged(node_t *p, bool v) noexcept
    {
        std::get<node_engaged_i>(*p) = v;
    }

    std::size_t &count_ref() noexcept
    {
        return std::get<core_count_i>(core_);
    }
    std::size_t count_ref() const noexcept
    {
        return std::get<core_count_i>(core_);
    }
    Compare &comp_ref() noexcept { return std::get<core_comp_i>(core_); }
    const Compare &comp_ref() const noexcept
    {
        return std::get<core_comp_i>(core_);
    }

    template <std::size_t I, class... Args>
    decltype(auto) engine(Args &&...args) const
    {
        return std::get<I>(core_)(core_, std::forward<Args>(args)...);
    }

    template <std::size_t I, class Core, class... Args>
    static decltype(auto) engine_on(Core &c, Args &&...args)
    {
        return std::get<I>(c)(c, std::forward<Args>(args)...);
    }

    template <class C, class = void>
    struct is_transparent : std::false_type
    {
    };
    template <class C>
    struct is_transparent<C, std::void_t<typename C::is_transparent>>
        : std::true_type
    {
    };

    template <class V, bool IsMapped>
    struct mapped_type_of
    {
        using type = V;
    };
    template <class V>
    struct mapped_type_of<V, true>
    {
        using type = typename V::second_type;
    };

    struct map_value_compare
    {
        Compare comp_;

        constexpr explicit map_value_compare(Compare comp)
            : comp_(std::move(comp)) {}

    public:
        using result_type = bool;

        constexpr bool operator()(const Value &x, const Value &y) const
        {
            return comp_(KeyOfValue{}(x), KeyOfValue{}(y));
        }
    };

public:
    // --- types -------------------------------------------------------------
    using key_type = Key;
    using mapped_type = typename mapped_type_of<Value, Mapped>::type;
    using value_type = Value;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;
    using key_compare = Compare;
    using value_compare =
        std::conditional_t<Mapped, map_value_compare, Compare>;
    using allocator_type = Allocator;
    using reference = value_type &;
    using const_reference = const value_type &;
    using pointer = typename std::allocator_traits<Allocator>::pointer;
    using const_pointer =
        typename std::allocator_traits<Allocator>::const_pointer;

private:
    template <bool Const>
    class iterator_impl
    {
        template <bool>
        friend class iterator_impl;
        friend class basic_ordered_tree;

        node_t *p_ = nullptr;
        next_fn next_ = nullptr;
        prev_fn prev_ = nullptr;

        constexpr iterator_impl(node_t *p, next_fn n,
                                prev_fn pr) noexcept
            : p_(p), next_(n), prev_(pr) {}

    public:
        using iterator_category = std::bidirectional_iterator_tag;
        using iterator_concept = std::bidirectional_iterator_tag;
        using value_type = Value;
        using difference_type = std::ptrdiff_t;
        using pointer =
            std::conditional_t<Const, const value_type *, value_type *>;
        using reference =
            std::conditional_t<Const, const value_type &, value_type &>;

        constexpr iterator_impl() noexcept = default;

        template <bool C>
            requires(Const || !C)
        constexpr iterator_impl(const iterator_impl<C> &other) noexcept
            : p_(other.p_), next_(other.next_), prev_(other.prev_)
        {
        }

        constexpr reference operator*() const noexcept
        {
            return *value_ptr(p_);
        }
        constexpr pointer operator->() const noexcept
        {
            return value_ptr(p_);
        }

        constexpr iterator_impl &operator++() noexcept
        {
            p_ = next_(p_);
            return *this;
        }
        constexpr iterator_impl operator++(int) noexcept
        {
            auto tmp = *this;
            ++*this;
            return tmp;
        }
        constexpr iterator_impl &operator--() noexcept
        {
            p_ = prev_(p_);
            return *this;
        }
        constexpr iterator_impl operator--(int) noexcept
        {
            auto tmp = *this;
            --*this;
            return tmp;
        }

        friend constexpr bool operator==(const iterator_impl &a,
                                         const iterator_impl &b) noexcept
        {
            return a.p_ == b.p_;
        }
    };

public:
    using iterator =
        std::conditional_t<Mapped, iterator_impl<false>, iterator_impl<true>>;
    using const_iterator = iterator_impl<true>;
    using reverse_iterator = std::reverse_iterator<iterator>;
    using const_reverse_iterator = std::reverse_iterator<const_iterator>;

    // --- node handle -------------------------------------------------------
    class node_type
    {
        friend class basic_ordered_tree;

    public:
        using allocator_type = typename basic_ordered_tree::allocator_type;
        using key_type = typename basic_ordered_tree::key_type;
        using mapped_type = typename basic_ordered_tree::mapped_type;
        using value_type = typename basic_ordered_tree::value_type;

    private:
        node_t *p_ = nullptr;
        allocator_type alloc_{};

        constexpr explicit node_type(const allocator_type &a)
            : p_(nullptr), alloc_(a) {}

        constexpr node_type(node_t *p, const allocator_type &a)
            : p_(p), alloc_(a) {}

        void reset() noexcept
        {
            if (p_)
            {
                destroy_node_with(alloc_, p_);
                p_ = nullptr;
            }
        }

        static void destroy_node_with(allocator_type &a, node_t *p) noexcept
        {
            if (engaged(p))
            {
                value_allocator_type va(a);
                value_alloc_traits::destroy(va, value_ptr(p));
            }
            node_allocator_type na(a);
            node_alloc_traits::destroy(na, p);
            node_alloc_traits::deallocate(
                na, std::pointer_traits<typename node_alloc_traits::pointer>::pointer_to(*p),
                1);
        }

    public:
        constexpr node_type() noexcept = default;
        node_type(const node_type &) = delete;
        node_type &operator=(const node_type &) = delete;

        constexpr node_type(node_type &&other) noexcept(
            std::is_nothrow_move_constructible_v<allocator_type>)
            : p_(std::exchange(other.p_, nullptr)),
              alloc_(std::move(other.alloc_)) {}

        constexpr node_type &operator=(node_type &&other) noexcept(
            std::is_nothrow_move_assignable_v<allocator_type>)
        {
            if (this != &other)
            {
                reset();
                p_ = std::exchange(other.p_, nullptr);
                alloc_ = std::move(other.alloc_);
            }
            return *this;
        }

        ~node_type() { reset(); }

        constexpr bool empty() const noexcept
        {
            return p_ == nullptr;
        }
        explicit constexpr operator bool() const noexcept
        {
            return p_ != nullptr;
        }

        constexpr allocator_type get_allocator() const
        {
            return alloc_;
        }

        constexpr value_type &value() const noexcept
        {
            return *value_ptr(p_);
        }

        constexpr key_type &key() const noexcept
            requires Mapped
        {
            return const_cast<key_type &>(KeyOfValue{}(*value_ptr(p_)));
        }

        constexpr mapped_type &mapped() const noexcept
            requires Mapped
        {
            return value_ptr(p_)->second;
        }

        constexpr void swap(node_type &other) noexcept(
            std::allocator_traits<
                allocator_type>::propagate_on_container_swap::value ||
            std::allocator_traits<allocator_type>::is_always_equal::value)
        {
            using std::swap;
            swap(p_, other.p_);
            if constexpr (std::allocator_traits<
                              allocator_type>::propagate_on_container_swap::value ||
                          std::allocator_traits<
                              allocator_type>::is_always_equal::value)
            {
                swap(alloc_, other.alloc_);
            }
            else if (!p_ || !other.p_ || alloc_ == other.alloc_)
            {
                swap(alloc_, other.alloc_);
            }
        }

        friend constexpr void swap(node_type &a, node_type &b) noexcept(
            noexcept(a.swap(b)))
        {
            a.swap(b);
        }
    };

    struct insert_return_type
    {
        iterator position;
        bool inserted;
        node_type node;
    };

    // For unique containers the standard insert() return type is exactly
    // std::pair<iterator,bool>, not a look-alike.
    using value_insert_return =
        std::conditional_t<AllowDuplicates, iterator,
                           std::pair<iterator, bool>>;
    using node_insert_return =
        std::conditional_t<AllowDuplicates, iterator, insert_return_type>;

private:
    mutable core_t core_;
    [[no_unique_address]] allocator_type alloc_;

    struct init_tag final
    {
    };

    constexpr basic_ordered_tree(init_tag, Compare comp, Allocator alloc)
        : core_(make_stl_toolkit.template operator()<
                Key, Value, KeyOfValue, Compare, AllowDuplicates>(
              std::move(comp))),
          alloc_(std::move(alloc))
    {
        engine<core_reset_i>();
    }

    // --- allocator-aware physical node lifetime ---------------------------
    node_t *allocate_node_storage()
    {
        node_allocator_type na(alloc_);
        auto fancy = node_alloc_traits::allocate(na, 1);
        node_t *p = std::to_address(fancy);
        try
        {
            node_alloc_traits::construct(na, p);
        }
        catch (...)
        {
            node_alloc_traits::deallocate(na, fancy, 1);
            throw;
        }
        return p;
    }

    void free_node_storage(node_t *p) noexcept
    {
        node_allocator_type na(alloc_);
        node_alloc_traits::destroy(na, p);
        node_alloc_traits::deallocate(
            na,
            std::pointer_traits<
                typename node_alloc_traits::pointer>::pointer_to(*p),
            1);
    }

    static void destroy_node_with(allocator_type &a, node_t *p) noexcept
    {
        if (engaged(p))
        {
            value_allocator_type va(a);
            value_alloc_traits::destroy(va, value_ptr(p));
        }
        node_allocator_type na(a);
        node_alloc_traits::destroy(na, p);
        node_alloc_traits::deallocate(
            na,
            std::pointer_traits<
                typename node_alloc_traits::pointer>::pointer_to(*p),
            1);
    }

    void destroy_node(node_t *p) noexcept { destroy_node_with(alloc_, p); }

    template <class... Args>
    node_t *create_node(Args &&...args)
    {
        node_t *p = allocate_node_storage();
        try
        {
            value_allocator_type va(alloc_);
            value_alloc_traits::construct(
                va, value_ptr(p), std::forward<Args>(args)...);
            set_engaged(p, true);
        }
        catch (...)
        {
            free_node_storage(p);
            throw;
        }
        return p;
    }

    void clear_nodes() noexcept
    {
        if (count_ref() != 0)
        {
            node_t *r = engine<core_root_node_i>();
            engine<core_clear_tree_i>(r, [this](node_t *p)
                                      { destroy_node(p); });
        }
        engine<core_reset_i>();
    }

    template <bool Move>
    void clone_from(basic_ordered_tree &other)
    {
        auto create = [this](node_t *src) -> node_t *
        {
            if constexpr (Move)
            {
                return create_node(std::move(*value_ptr(src)));
            }
            else
            {
                return create_node(*value_ptr(src));
            }
        };
        auto destroy = [this](node_t *p)
        { destroy_node(p); };
        node_t *r = engine<core_clone_tree_i>(
            engine_on<core_root_node_i>(other.core_), create, destroy);
        engine<core_adopt_i>(r, other.count_ref());
    }

    template <bool Const>
    constexpr iterator_impl<Const> iter_at(node_t *p) const noexcept
    {
        return iterator_impl<Const>(
            p, std::get<core_next_i>(core_), std::get<core_prev_i>(core_));
    }

    template <bool Const>
    constexpr iterator_impl<Const> begin_impl() const noexcept
    {
        return iter_at<Const>(engine<core_first_node_i>());
    }

    template <bool Const>
    constexpr iterator_impl<Const> end_impl() const noexcept
    {
        return iter_at<Const>(engine<core_end_node_i>());
    }

    template <class... Args>
    value_insert_return emplace_impl(Args &&...args)
    {
        node_t *p = create_node(std::forward<Args>(args)...);
        node_t *pos = nullptr;
        bool inserted = false;
        try
        {
            auto result = engine<core_insert_node_i>(p);
            pos = result.first;
            inserted = result.second;
        }
        catch (...)
        {
            destroy_node(p);
            throw;
        }
        if (!inserted)
        {
            destroy_node(p);
        }
        if constexpr (AllowDuplicates)
        {
            return iter_at<Mapped ? false : true>(pos);
        }
        else
        {
            return {iter_at<false>(pos), inserted};
        }
    }

    template <class... Args>
    iterator emplace_hint_impl(const_iterator hint, Args &&...args)
    {
        node_t *p = create_node(std::forward<Args>(args)...);
        node_t *pos = nullptr;
        bool inserted = false;
        try
        {
            auto result = engine<core_insert_node_hint_i>(p, hint.p_);
            pos = result.first;
            inserted = result.second;
        }
        catch (...)
        {
            destroy_node(p);
            throw;
        }
        if (!inserted)
        {
            destroy_node(p);
        }
        return iter_at<Mapped ? false : true>(pos);
    }

    template <bool Const, class K>
    iterator_impl<Const> lower_bound_impl(const K &k) const
    {
        return iter_at<Const>(engine<core_lower_bound_i>(k));
    }

    template <bool Const, class K>
    iterator_impl<Const> upper_bound_impl(const K &k) const
    {
        return iter_at<Const>(engine<core_upper_bound_i>(k));
    }

    template <bool Const, class K>
    iterator_impl<Const> find_impl(const K &k) const
    {
        return iter_at<Const>(engine<core_find_node_i>(k));
    }

    template <bool Const, class K>
    std::pair<iterator_impl<Const>, iterator_impl<Const>>
    equal_range_impl(const K &k) const
    {
        return {iter_at<Const>(engine<core_lower_bound_i>(k)),
                iter_at<Const>(engine<core_upper_bound_i>(k))};
    }

    template <class K>
    size_type count_impl(const K &k) const
    {
        node_t *first = engine<core_lower_bound_i>(k);
        node_t *last = engine<core_upper_bound_i>(k);
        size_type n = 0;
        while (first != last)
        {
            ++n;
            first = std::get<core_next_i>(core_)(first);
        }
        return n;
    }

    template <class K>
    size_type erase_key_impl(const K &k)
    {
        node_t *first = engine<core_lower_bound_i>(k);
        node_t *last = engine<core_upper_bound_i>(k);
        size_type n = 0;
        while (first != last)
        {
            node_t *next = std::get<core_next_i>(core_)(first);
            engine<core_detach_i>(first);
            destroy_node(first);
            first = next;
            ++n;
        }
        return n;
    }

public:
    // --- construction ------------------------------------------------------
    basic_ordered_tree() : basic_ordered_tree(init_tag{}, Compare{},
                                              Allocator{}) {}

    explicit basic_ordered_tree(const Compare &comp,
                                const Allocator &alloc = Allocator())
        : basic_ordered_tree(init_tag{}, comp, alloc) {}

    explicit basic_ordered_tree(const Allocator &alloc)
        : basic_ordered_tree(init_tag{}, Compare{}, alloc) {}

    template <std::input_iterator InputIt>
    basic_ordered_tree(InputIt first, InputIt last,
                       const Compare &comp = Compare(),
                       const Allocator &alloc = Allocator())
        : basic_ordered_tree(init_tag{}, comp, alloc)
    {
        try
        {
            insert(first, last);
        }
        catch (...)
        {
            clear_nodes();
            throw;
        }
    }

    template <std::input_iterator InputIt>
    basic_ordered_tree(InputIt first, InputIt last, const Allocator &alloc)
        : basic_ordered_tree(init_tag{}, Compare{}, alloc)
    {
        try
        {
            insert(first, last);
        }
        catch (...)
        {
            clear_nodes();
            throw;
        }
    }

    basic_ordered_tree(std::initializer_list<value_type> init,
                       const Compare &comp = Compare(),
                       const Allocator &alloc = Allocator())
        : basic_ordered_tree(init_tag{}, comp, alloc)
    {
        try
        {
            insert(init.begin(), init.end());
        }
        catch (...)
        {
            clear_nodes();
            throw;
        }
    }

    basic_ordered_tree(std::initializer_list<value_type> init,
                       const Allocator &alloc)
        : basic_ordered_tree(init_tag{}, Compare{}, alloc)
    {
        try
        {
            insert(init.begin(), init.end());
        }
        catch (...)
        {
            clear_nodes();
            throw;
        }
    }

    template <std::ranges::input_range R>
        requires std::convertible_to<std::ranges::range_reference_t<R>,
                                     value_type>
    basic_ordered_tree(std::from_range_t, R &&rg,
                       const Compare &comp = Compare(),
                       const Allocator &alloc = Allocator())
        : basic_ordered_tree(init_tag{}, comp, alloc)
    {
        try
        {
            insert_range(std::forward<R>(rg));
        }
        catch (...)
        {
            clear_nodes();
            throw;
        }
    }

    template <std::ranges::input_range R>
        requires std::convertible_to<std::ranges::range_reference_t<R>,
                                     value_type>
    basic_ordered_tree(std::from_range_t, R &&rg, const Allocator &alloc)
        : basic_ordered_tree(init_tag{}, Compare{}, alloc)
    {
        try
        {
            insert_range(std::forward<R>(rg));
        }
        catch (...)
        {
            clear_nodes();
            throw;
        }
    }

    basic_ordered_tree(const basic_ordered_tree &other)
        : basic_ordered_tree(
              init_tag{}, other.comp_ref(),
              std::allocator_traits<
                  Allocator>::select_on_container_copy_construction(other.alloc_))
    {
        clone_from<false>(const_cast<basic_ordered_tree &>(other));
    }

    basic_ordered_tree(basic_ordered_tree &&other) noexcept(
        std::allocator_traits<Allocator>::is_always_equal::value &&
        std::is_nothrow_move_constructible_v<Compare>)
        : basic_ordered_tree(init_tag{}, std::move(other.comp_ref()),
                             std::move(other.alloc_))
    {
        engine<core_take_from_i>(other.core_);
    }

    basic_ordered_tree(const basic_ordered_tree &other,
                       const std::type_identity_t<Allocator> &alloc)
        : basic_ordered_tree(init_tag{}, other.comp_ref(), alloc)
    {
        clone_from<false>(const_cast<basic_ordered_tree &>(other));
    }

    basic_ordered_tree(basic_ordered_tree &&other,
                       const std::type_identity_t<Allocator> &alloc)
        : basic_ordered_tree(init_tag{}, std::move(other.comp_ref()), alloc)
    {
        if (alloc_ == other.alloc_)
        {
            engine<core_take_from_i>(other.core_);
        }
        else
        {
            clone_from<true>(other);
            other.clear();
        }
    }

    ~basic_ordered_tree() { clear_nodes(); }

    // --- assignment --------------------------------------------------------
    basic_ordered_tree &operator=(const basic_ordered_tree &other)
    {
        if (this == &other)
        {
            return *this;
        }
        using pocca = typename std::allocator_traits<
            Allocator>::propagate_on_container_copy_assignment;
        comp_ref() = other.comp_ref();
        if constexpr (pocca::value)
        {
            if (!(alloc_ == other.alloc_))
            {
                clear_nodes();
            }
            alloc_ = other.alloc_;
        }
        clear_nodes();
        clone_from<false>(const_cast<basic_ordered_tree &>(other));
        return *this;
    }

    basic_ordered_tree &operator=(basic_ordered_tree &&other) noexcept(
        (std::allocator_traits<
             Allocator>::propagate_on_container_move_assignment::value ||
         std::allocator_traits<Allocator>::is_always_equal::value) &&
        std::is_nothrow_move_assignable_v<Compare>)
    {
        if (this == &other)
        {
            return *this;
        }
        using pocma = typename std::allocator_traits<
            Allocator>::propagate_on_container_move_assignment;
        comp_ref() = std::move(other.comp_ref());
        if constexpr (pocma::value)
        {
            if (!(alloc_ == other.alloc_))
            {
                clear_nodes();
            }
            else
            {
                clear_nodes();
            }
            alloc_ = std::move(other.alloc_);
            engine<core_take_from_i>(other.core_);
        }
        else
        {
            if (alloc_ == other.alloc_)
            {
                clear_nodes();
                engine<core_take_from_i>(other.core_);
            }
            else
            {
                clear_nodes();
                clone_from<true>(other);
                other.clear();
            }
        }
        return *this;
    }

    basic_ordered_tree &operator=(std::initializer_list<value_type> init)
    {
        clear();
        insert(init.begin(), init.end());
        return *this;
    }

    // --- allocator ---------------------------------------------------------
    allocator_type get_allocator() const noexcept
    {
        return alloc_;
    }

    // --- iterators ---------------------------------------------------------
    iterator begin() noexcept { return begin_impl<Mapped ? false : true>(); }
    const_iterator begin() const noexcept { return begin_impl<true>(); }
    const_iterator cbegin() const noexcept { return begin_impl<true>(); }

    iterator end() noexcept { return end_impl<Mapped ? false : true>(); }
    const_iterator end() const noexcept { return end_impl<true>(); }
    const_iterator cend() const noexcept { return end_impl<true>(); }

    reverse_iterator rbegin() noexcept { return reverse_iterator(end()); }
    const_reverse_iterator rbegin() const noexcept
    {
        return const_reverse_iterator(end());
    }
    const_reverse_iterator crbegin() const noexcept
    {
        return const_reverse_iterator(cend());
    }

    reverse_iterator rend() noexcept { return reverse_iterator(begin()); }
    const_reverse_iterator rend() const noexcept
    {
        return const_reverse_iterator(begin());
    }
    const_reverse_iterator crend() const noexcept
    {
        return const_reverse_iterator(cbegin());
    }

    // --- capacity ----------------------------------------------------------
    [[nodiscard]] bool empty() const noexcept { return count_ref() == 0; }
    [[nodiscard]] size_type size() const noexcept { return count_ref(); }
    size_type max_size() const noexcept
    {
        node_allocator_type na(alloc_);
        return node_alloc_traits::max_size(na);
    }

    // --- modifiers ---------------------------------------------------------
    void clear() noexcept { clear_nodes(); }

    value_insert_return insert(const value_type &value)
    {
        return emplace_impl(value);
    }
    value_insert_return insert(value_type &&value)
    {
        return emplace_impl(std::move(value));
    }

    template <class P>
        requires(!std::same_as<std::remove_cvref_t<P>, value_type> &&
                 std::constructible_from<value_type, P>)
    value_insert_return insert(P &&value)
    {
        return emplace_impl(std::forward<P>(value));
    }

    iterator insert(const_iterator hint, const value_type &value)
    {
        return emplace_hint_impl(hint, value);
    }
    iterator insert(const_iterator hint, value_type &&value)
    {
        return emplace_hint_impl(hint, std::move(value));
    }

    template <class P>
        requires(!std::same_as<std::remove_cvref_t<P>, value_type> &&
                 std::constructible_from<value_type, P>)
    iterator insert(const_iterator hint, P &&value)
    {
        return emplace_hint_impl(hint, std::forward<P>(value));
    }

    template <std::input_iterator InputIt>
    void insert(InputIt first, InputIt last)
    {
        while (first != last)
        {
            insert(*first);
            ++first;
        }
    }

    void insert(std::initializer_list<value_type> init)
    {
        insert(init.begin(), init.end());
    }

    template <std::ranges::input_range R>
        requires std::convertible_to<std::ranges::range_reference_t<R>,
                                     value_type>
    void insert_range(R &&rg)
    {
        for (auto &&e : rg)
        {
            insert(std::forward<decltype(e)>(e));
        }
    }

    node_insert_return insert(node_type &&nh)
    {
        if (nh.empty())
        {
            if constexpr (AllowDuplicates)
            {
                return end();
            }
            else
            {
                return {end(), false, node_type{}};
            }
        }
        if constexpr (!AllowDuplicates)
        {
            const key_type &k = KeyOfValue{}(*value_ptr(nh.p_));
            auto it = find(k);
            if (it != end())
            {
                return {it, false, std::move(nh)};
            }
        }

        if (alloc_ == nh.alloc_)
        {
            node_t *p = nh.p_;
            auto [pos, inserted] = engine<core_insert_node_i>(p);
            if (inserted)
            {
                nh.p_ = nullptr;
            }
            if constexpr (AllowDuplicates)
            {
                return iter_at<Mapped ? false : true>(pos);
            }
            else
            {
                if (inserted)
                {
                    return {iter_at<false>(pos), true, node_type{}};
                }
                return {iter_at<false>(pos), false, std::move(nh)};
            }
        }
        else
        {
            node_t *p = create_node(std::move(*value_ptr(nh.p_)));
            node_t *pos = nullptr;
            bool inserted = false;
            try
            {
                auto result = engine<core_insert_node_i>(p);
                pos = result.first;
                inserted = result.second;
            }
            catch (...)
            {
                destroy_node(p);
                throw;
            }
            if (!inserted)
            {
                destroy_node(p);
                if constexpr (AllowDuplicates)
                {
                    return iter_at<Mapped ? false : true>(pos);
                }
                else
                {
                    return {iter_at<false>(pos), false, std::move(nh)};
                }
            }
            nh.reset();
            if constexpr (AllowDuplicates)
            {
                return iter_at<Mapped ? false : true>(pos);
            }
            else
            {
                return {iter_at<false>(pos), true, node_type{}};
            }
        }
    }

    node_insert_return insert(const node_type &nh)
    {
        if (nh.empty())
        {
            if constexpr (AllowDuplicates)
            {
                return end();
            }
            else
            {
                return {end(), false, node_type{}};
            }
        }
        if constexpr (!AllowDuplicates)
        {
            auto it = find(KeyOfValue{}(*value_ptr(nh.p_)));
            if (it != end())
            {
                return {it, false, node_type{}};
            }
        }
        node_t *p = create_node(*value_ptr(nh.p_));
        node_t *pos = nullptr;
        bool inserted = false;
        try
        {
            auto result = engine<core_insert_node_i>(p);
            pos = result.first;
            inserted = result.second;
        }
        catch (...)
        {
            destroy_node(p);
            throw;
        }
        if (!inserted)
        {
            destroy_node(p);
        }
        if constexpr (AllowDuplicates)
        {
            return iter_at<Mapped ? false : true>(pos);
        }
        else
        {
            return {iter_at<false>(pos), inserted, node_type{}};
        }
    }

    iterator insert(const_iterator hint, node_type &&nh)
    {
        if (nh.empty())
        {
            return end();
        }
        if (alloc_ == nh.alloc_)
        {
            node_t *p = nh.p_;
            auto result = engine<core_insert_node_hint_i>(p, hint.p_);
            if (result.second)
            {
                nh.p_ = nullptr;
            }
            return iter_at<Mapped ? false : true>(result.first);
        }
        if constexpr (!AllowDuplicates)
        {
            auto it = find(KeyOfValue{}(*value_ptr(nh.p_)));
            if (it != end())
            {
                return it;
            }
        }
        // Different allocator: node handles cannot transfer storage; move the
        // value into a node built by this container's allocator.
        auto r = insert(std::move(nh));
        if constexpr (AllowDuplicates)
        {
            return r;
        }
        else
        {
            return r.position;
        }
    }

    iterator insert(const_iterator, const node_type &nh)
    {
        if constexpr (AllowDuplicates)
        {
            return insert(nh);
        }
        else
        {
            auto r = insert(nh);
            return r.position;
        }
    }

    template <class... Args>
    value_insert_return emplace(Args &&...args)
    {
        return emplace_impl(std::forward<Args>(args)...);
    }

    template <class... Args>
    iterator emplace_hint(const_iterator hint, Args &&...args)
    {
        return emplace_hint_impl(hint, std::forward<Args>(args)...);
    }

    iterator erase(const_iterator pos)
    {
        node_t *p = pos.p_;
        node_t *n = std::get<core_next_i>(core_)(p);
        engine<core_detach_i>(p);
        destroy_node(p);
        return iter_at<Mapped ? false : true>(n);
    }

    iterator erase(iterator pos)
        requires Mapped
    {
        return erase(const_iterator(pos));
    }

    iterator erase(const_iterator first, const_iterator last)
    {
        node_t *stop = last.p_;
        node_t *cur = first.p_;
        while (cur != stop)
        {
            node_t *n = std::get<core_next_i>(core_)(cur);
            engine<core_detach_i>(cur);
            destroy_node(cur);
            cur = n;
        }
        return iter_at<Mapped ? false : true>(stop);
    }

    size_type erase(const key_type &key) { return erase_key_impl(key); }

    template <class K>
        requires(is_transparent<Compare>::value &&
                 !std::convertible_to<K, iterator> &&
                 !std::convertible_to<K, const_iterator>)
    size_type erase(K &&key)
    {
        return erase_key_impl(key);
    }

    void swap(basic_ordered_tree &other) noexcept(
        (std::allocator_traits<
             Allocator>::propagate_on_container_swap::value ||
         std::allocator_traits<Allocator>::is_always_equal::value) &&
        std::is_nothrow_swappable_v<Compare>)
    {
        if (this == &other)
        {
            return;
        }
        using std::swap;
        swap(comp_ref(), other.comp_ref());
        engine<core_swap_trees_i>(other.core_);
        if constexpr (std::allocator_traits<
                          Allocator>::propagate_on_container_swap::value)
        {
            swap(alloc_, other.alloc_);
        }
    }

    friend void swap(basic_ordered_tree &a, basic_ordered_tree &b) noexcept(
        noexcept(a.swap(b)))
    {
        a.swap(b);
    }

    // --- extraction / merge ------------------------------------------------
    node_type extract(const_iterator pos)
    {
        node_t *p = pos.p_;
        node_type result(alloc_);
        engine<core_detach_i>(p);
        result.p_ = p;
        return result;
    }

    node_type extract(iterator pos)
        requires Mapped
    {
        return extract(const_iterator(pos));
    }

    node_type extract(const key_type &key)
    {
        node_t *p = engine<core_find_node_i>(key);
        if (p == engine<core_end_node_i>())
        {
            return node_type{};
        }
        node_type result(alloc_);
        engine<core_detach_i>(p);
        result.p_ = p;
        return result;
    }

    template <class K>
        requires(is_transparent<Compare>::value &&
                 !std::convertible_to<K, iterator> &&
                 !std::convertible_to<K, const_iterator>)
    node_type extract(K &&key)
    {
        node_t *p = engine<core_find_node_i>(key);
        if (p == engine<core_end_node_i>())
        {
            return node_type{};
        }
        node_type result(alloc_);
        engine<core_detach_i>(p);
        result.p_ = p;
        return result;
    }

    void merge(basic_ordered_tree &source)
    {
        if (static_cast<const void *>(this) ==
            static_cast<const void *>(std::addressof(source)))
        {
            return;
        }
        auto it = source.begin();
        auto last = source.end();
        while (it != last)
        {
            auto cur = it++;
            auto nh = source.extract(cur);
            if constexpr (AllowDuplicates)
            {
                insert(std::move(nh));
            }
            else
            {
                auto r = insert(std::move(nh));
                if (!r.inserted)
                {
                    source.insert(std::move(r.node));
                }
            }
        }
    }

    void merge(basic_ordered_tree &&source) { merge(source); }

    // --- lookup ------------------------------------------------------------
    size_type count(const key_type &key) const { return count_impl(key); }
    template <class K>
        requires is_transparent<Compare>::value
    size_type count(const K &key) const
    {
        return count_impl(key);
    }

    iterator find(const key_type &key)
    {
        return find_impl<Mapped ? false : true>(key);
    }
    const_iterator find(const key_type &key) const
    {
        return find_impl<true>(key);
    }
    template <class K>
        requires is_transparent<Compare>::value
    iterator find(const K &key)
    {
        return find_impl<Mapped ? false : true>(key);
    }
    template <class K>
        requires is_transparent<Compare>::value
    const_iterator find(const K &key) const
    {
        return find_impl<true>(key);
    }

    bool contains(const key_type &key) const { return find(key) != end(); }
    template <class K>
        requires is_transparent<Compare>::value
    bool contains(const K &key) const
    {
        return find(key) != end();
    }

    std::pair<iterator, iterator> equal_range(const key_type &key)
    {
        return equal_range_impl<Mapped ? false : true>(key);
    }
    std::pair<const_iterator, const_iterator>
    equal_range(const key_type &key) const
    {
        return equal_range_impl<true>(key);
    }
    template <class K>
        requires is_transparent<Compare>::value
    std::pair<iterator, iterator> equal_range(const K &key)
    {
        return equal_range_impl<Mapped ? false : true>(key);
    }
    template <class K>
        requires is_transparent<Compare>::value
    std::pair<const_iterator, const_iterator> equal_range(const K &key) const
    {
        return equal_range_impl<true>(key);
    }

    iterator lower_bound(const key_type &key)
    {
        return lower_bound_impl<Mapped ? false : true>(key);
    }
    const_iterator lower_bound(const key_type &key) const
    {
        return lower_bound_impl<true>(key);
    }
    template <class K>
        requires is_transparent<Compare>::value
    iterator lower_bound(const K &key)
    {
        return lower_bound_impl<Mapped ? false : true>(key);
    }
    template <class K>
        requires is_transparent<Compare>::value
    const_iterator lower_bound(const K &key) const
    {
        return lower_bound_impl<true>(key);
    }

    iterator upper_bound(const key_type &key)
    {
        return upper_bound_impl<Mapped ? false : true>(key);
    }
    const_iterator upper_bound(const key_type &key) const
    {
        return upper_bound_impl<true>(key);
    }
    template <class K>
        requires is_transparent<Compare>::value
    iterator upper_bound(const K &key)
    {
        return upper_bound_impl<Mapped ? false : true>(key);
    }
    template <class K>
        requires is_transparent<Compare>::value
    const_iterator upper_bound(const K &key) const
    {
        return upper_bound_impl<true>(key);
    }

    // --- observers ---------------------------------------------------------
    key_compare key_comp() const { return comp_ref(); }

    value_compare value_comp() const
    {
        if constexpr (Mapped)
        {
            return value_compare(comp_ref());
        }
        else
        {
            return comp_ref();
        }
    }

    // --- map-specific element access --------------------------------------
    mapped_type &at(const key_type &key)
        requires Mapped
    {
        auto it = find(key);
        if (it == end())
        {
            throw std::out_of_range("basic_ordered_tree::at");
        }
        return it->second;
    }

    const mapped_type &at(const key_type &key) const
        requires Mapped
    {
        auto it = find(key);
        if (it == end())
        {
            throw std::out_of_range("basic_ordered_tree::at");
        }
        return it->second;
    }

    mapped_type &operator[](const key_type &key)
        requires(Mapped && !AllowDuplicates)
    {
        return try_emplace(key).first->second;
    }

    mapped_type &operator[](key_type &&key)
        requires(Mapped && !AllowDuplicates)
    {
        return try_emplace(std::move(key)).first->second;
    }

    template <class... Args>
    std::pair<iterator, bool> try_emplace(const key_type &key,
                                          Args &&...args)
        requires(Mapped && !AllowDuplicates)
    {
        auto it = find(key);
        if (it != end())
        {
            return {it, false};
        }
        node_t *p = create_node(
            std::piecewise_construct,
            std::forward_as_tuple(key),
            std::forward_as_tuple(std::forward<Args>(args)...));
        node_t *pos = nullptr;
        bool inserted = false;
        try
        {
            auto result = engine<core_insert_node_i>(p);
            pos = result.first;
            inserted = result.second;
        }
        catch (...)
        {
            destroy_node(p);
            throw;
        }
        if (!inserted)
        {
            destroy_node(p);
        }
        return {iter_at<false>(pos), inserted};
    }

    template <class... Args>
    std::pair<iterator, bool> try_emplace(key_type &&key, Args &&...args)
        requires(Mapped && !AllowDuplicates)
    {
        auto it = find(key);
        if (it != end())
        {
            return {it, false};
        }
        node_t *p = create_node(
            std::piecewise_construct,
            std::forward_as_tuple(std::move(key)),
            std::forward_as_tuple(std::forward<Args>(args)...));
        node_t *pos = nullptr;
        bool inserted = false;
        try
        {
            auto result = engine<core_insert_node_i>(p);
            pos = result.first;
            inserted = result.second;
        }
        catch (...)
        {
            destroy_node(p);
            throw;
        }
        if (!inserted)
        {
            destroy_node(p);
        }
        return {iter_at<false>(pos), inserted};
    }

    template <class... Args>
    iterator try_emplace(const_iterator, const key_type &key, Args &&...args)
        requires(Mapped && !AllowDuplicates)
    {
        return try_emplace(key, std::forward<Args>(args)...).first;
    }

    template <class... Args>
    iterator try_emplace(const_iterator, key_type &&key, Args &&...args)
        requires(Mapped && !AllowDuplicates)
    {
        return try_emplace(std::move(key), std::forward<Args>(args)...).first;
    }

    template <class M>
    std::pair<iterator, bool> insert_or_assign(const key_type &key, M &&obj)
        requires(Mapped && !AllowDuplicates)
    {
        auto it = find(key);
        if (it != end())
        {
            it->second = std::forward<M>(obj);
            return {it, false};
        }
        node_t *p = create_node(
            std::piecewise_construct, std::forward_as_tuple(key),
            std::forward_as_tuple(std::forward<M>(obj)));
        node_t *pos = nullptr;
        bool inserted = false;
        try
        {
            auto result = engine<core_insert_node_i>(p);
            pos = result.first;
            inserted = result.second;
        }
        catch (...)
        {
            destroy_node(p);
            throw;
        }
        if (!inserted)
        {
            destroy_node(p);
        }
        return {iter_at<false>(pos), inserted};
    }

    template <class M>
    std::pair<iterator, bool> insert_or_assign(key_type &&key, M &&obj)
        requires(Mapped && !AllowDuplicates)
    {
        auto it = find(key);
        if (it != end())
        {
            it->second = std::forward<M>(obj);
            return {it, false};
        }
        node_t *p = create_node(
            std::piecewise_construct,
            std::forward_as_tuple(std::move(key)),
            std::forward_as_tuple(std::forward<M>(obj)));
        auto [pos, inserted] = engine<core_insert_node_i>(p);
        if (!inserted)
        {
            destroy_node(p);
        }
        return {iter_at<false>(pos), inserted};
    }

    template <class M>
    iterator insert_or_assign(const_iterator, const key_type &key, M &&obj)
        requires(Mapped && !AllowDuplicates)
    {
        return insert_or_assign(key, std::forward<M>(obj)).first;
    }

    template <class M>
    iterator insert_or_assign(const_iterator, key_type &&key, M &&obj)
        requires(Mapped && !AllowDuplicates)
    {
        return insert_or_assign(std::move(key), std::forward<M>(obj)).first;
    }

    // --- relational operators ---------------------------------------------
    friend bool operator==(const basic_ordered_tree &a,
                           const basic_ordered_tree &b)
    {
        return a.size() == b.size() &&
               std::equal(a.begin(), a.end(), b.begin());
    }

    friend auto operator<=>(const basic_ordered_tree &a,
                            const basic_ordered_tree &b)
    {
        return std::lexicographical_compare_three_way(
            a.begin(), a.end(), b.begin(), b.end(),
            std::compare_three_way{});
    }
};

// ---------------------------------------------------------------------------
// Public standard facades.  They are ordinary global class templates and add
// no data: all semantics and the private closure-core member live in the base.
// ---------------------------------------------------------------------------
template <class Key, class Compare = std::less<Key>,
          class Allocator = std::allocator<Key>>
class set
    : public basic_ordered_tree<Key, Key, std::identity, Compare, Allocator,
                                false, false>
{
    using base = basic_ordered_tree<Key, Key, std::identity, Compare,
                                    Allocator, false, false>;

public:
    using base::base;
    using value_type = typename base::value_type;
    set() = default;
    set(const set &) = default;
    set(set &&) = default;
    set &operator=(const set &) = default;
    set &operator=(set &&) = default;
    set &operator=(std::initializer_list<value_type> init)
    {
        base::operator=(init);
        return *this;
    }
    set(const set &other, const std::type_identity_t<Allocator> &alloc)
        : base(static_cast<const base &>(other), alloc) {}
    set(set &&other, const std::type_identity_t<Allocator> &alloc)
        : base(static_cast<base &&>(other), alloc) {}
};

template <class Key, class Compare = std::less<Key>,
          class Allocator = std::allocator<Key>>
class multiset
    : public basic_ordered_tree<Key, Key, std::identity, Compare, Allocator,
                                true, false>
{
    using base = basic_ordered_tree<Key, Key, std::identity, Compare,
                                    Allocator, true, false>;

public:
    using base::base;
    using value_type = typename base::value_type;
    multiset() = default;
    multiset(const multiset &) = default;
    multiset(multiset &&) = default;
    multiset &operator=(const multiset &) = default;
    multiset &operator=(multiset &&) = default;
    multiset &operator=(std::initializer_list<value_type> init)
    {
        base::operator=(init);
        return *this;
    }
    multiset(const multiset &other,
             const std::type_identity_t<Allocator> &alloc)
        : base(static_cast<const base &>(other), alloc) {}
    multiset(multiset &&other, const std::type_identity_t<Allocator> &alloc)
        : base(static_cast<base &&>(other), alloc) {}
};

template <class Key, class T, class Compare = std::less<Key>,
          class Allocator = std::allocator<std::pair<const Key, T>>>
class map
    : public basic_ordered_tree<Key, std::pair<const Key, T>,
                                closure_key_first, Compare, Allocator, false,
                                true>
{
    using base = basic_ordered_tree<Key, std::pair<const Key, T>,
                                    closure_key_first, Compare, Allocator,
                                    false, true>;

public:
    using base::base;
    using value_type = typename base::value_type;
    map() = default;
    map(const map &) = default;
    map(map &&) = default;
    map &operator=(const map &) = default;
    map &operator=(map &&) = default;
    map &operator=(std::initializer_list<value_type> init)
    {
        base::operator=(init);
        return *this;
    }
    map(const map &other, const std::type_identity_t<Allocator> &alloc)
        : base(static_cast<const base &>(other), alloc) {}
    map(map &&other, const std::type_identity_t<Allocator> &alloc)
        : base(static_cast<base &&>(other), alloc) {}
};

template <class Key, class T, class Compare = std::less<Key>,
          class Allocator = std::allocator<std::pair<const Key, T>>>
class multimap
    : public basic_ordered_tree<Key, std::pair<const Key, T>,
                                closure_key_first, Compare, Allocator, true,
                                true>
{
    using base = basic_ordered_tree<Key, std::pair<const Key, T>,
                                    closure_key_first, Compare, Allocator, true,
                                    true>;

public:
    using base::base;
    using value_type = typename base::value_type;
    multimap() = default;
    multimap(const multimap &) = default;
    multimap(multimap &&) = default;
    multimap &operator=(const multimap &) = default;
    multimap &operator=(multimap &&) = default;
    multimap &operator=(std::initializer_list<value_type> init)
    {
        base::operator=(init);
        return *this;
    }
    multimap(const multimap &other,
             const std::type_identity_t<Allocator> &alloc)
        : base(static_cast<const base &>(other), alloc) {}
    multimap(multimap &&other, const std::type_identity_t<Allocator> &alloc)
        : base(static_cast<base &&>(other), alloc) {}
};

namespace
{
    template <class A>
    concept closure_allocator_like = requires(A &a, std::size_t n) {
        typename A::value_type;
        a.allocate(n);
    };
} // namespace

// --- deduction guides ------------------------------------------------------
template <std::input_iterator It, class Key = std::iter_value_t<It>,
          class Compare = std::less<Key>,
          class Allocator = std::allocator<Key>>
    requires(!closure_allocator_like<Compare>)
set(It, It, Compare = Compare(), Allocator = Allocator())
    -> set<Key, Compare, Allocator>;

template <std::input_iterator It, class Key = std::iter_value_t<It>,
          closure_allocator_like Allocator>
set(It, It, Allocator) -> set<Key, std::less<Key>, Allocator>;

template <class Key, class Compare = std::less<Key>,
          class Allocator = std::allocator<Key>>
    requires(!closure_allocator_like<Compare>)
set(std::initializer_list<Key>, Compare = Compare(),
    Allocator = Allocator()) -> set<Key, Compare, Allocator>;

template <class Key, closure_allocator_like Allocator>
set(std::initializer_list<Key>, Allocator) -> set<Key, std::less<Key>, Allocator>;

template <std::ranges::input_range R, class Key = std::ranges::range_value_t<R>,
          class Compare = std::less<Key>,
          class Allocator = std::allocator<Key>>
    requires(!closure_allocator_like<Compare>)
set(std::from_range_t, R &&, Compare = Compare(), Allocator = Allocator())
    -> set<Key, Compare, Allocator>;

template <std::ranges::input_range R, class Key = std::ranges::range_value_t<R>,
          closure_allocator_like Allocator>
set(std::from_range_t, R &&, Allocator)
    -> set<Key, std::less<Key>, Allocator>;

template <std::input_iterator It, class Key = std::iter_value_t<It>,
          class Compare = std::less<Key>,
          class Allocator = std::allocator<Key>>
    requires(!closure_allocator_like<Compare>)
multiset(It, It, Compare = Compare(), Allocator = Allocator())
    -> multiset<Key, Compare, Allocator>;

template <std::input_iterator It, class Key = std::iter_value_t<It>,
          closure_allocator_like Allocator>
multiset(It, It, Allocator) -> multiset<Key, std::less<Key>, Allocator>;

template <class Key, class Compare = std::less<Key>,
          class Allocator = std::allocator<Key>>
    requires(!closure_allocator_like<Compare>)
multiset(std::initializer_list<Key>, Compare = Compare(),
         Allocator = Allocator()) -> multiset<Key, Compare, Allocator>;

template <class Key, closure_allocator_like Allocator>
multiset(std::initializer_list<Key>, Allocator)
    -> multiset<Key, std::less<Key>, Allocator>;

template <std::ranges::input_range R, class Key = std::ranges::range_value_t<R>,
          class Compare = std::less<Key>,
          class Allocator = std::allocator<Key>>
    requires(!closure_allocator_like<Compare>)
multiset(std::from_range_t, R &&, Compare = Compare(),
         Allocator = Allocator()) -> multiset<Key, Compare, Allocator>;

template <std::ranges::input_range R, class Key = std::ranges::range_value_t<R>,
          closure_allocator_like Allocator>
multiset(std::from_range_t, R &&, Allocator)
    -> multiset<Key, std::less<Key>, Allocator>;

template <class Pair, class = void>
struct closure_pair_info
{
};

template <class Pair>
struct closure_pair_info<
    Pair, std::void_t<typename Pair::first_type, typename Pair::second_type>>
{
    using key_type = std::remove_const_t<typename Pair::first_type>;
    using mapped_type = typename Pair::second_type;
};

template <std::input_iterator It,
          class Info = closure_pair_info<std::iter_value_t<It>>,
          class Key = typename Info::key_type,
          class T = typename Info::mapped_type,
          class Compare = std::less<Key>,
          class Allocator = std::allocator<std::pair<const Key, T>>>
    requires(!closure_allocator_like<Compare>)
map(It, It, Compare = Compare(), Allocator = Allocator())
    -> map<Key, T, Compare, Allocator>;

template <std::input_iterator It,
          class Info = closure_pair_info<std::iter_value_t<It>>,
          class Key = typename Info::key_type,
          class T = typename Info::mapped_type,
          closure_allocator_like Allocator>
map(It, It, Allocator) -> map<Key, T, std::less<Key>, Allocator>;

template <class Key, class T, class Compare = std::less<Key>,
          class Allocator = std::allocator<std::pair<const Key, T>>>
    requires(!closure_allocator_like<Compare>)
map(std::initializer_list<std::pair<Key, T>>, Compare = Compare(),
    Allocator = Allocator()) -> map<Key, T, Compare, Allocator>;

template <class Key, class T, closure_allocator_like Allocator>
map(std::initializer_list<std::pair<Key, T>>, Allocator)
    -> map<Key, T, std::less<Key>, Allocator>;

template <std::ranges::input_range R,
          class Info = closure_pair_info<std::ranges::range_value_t<R>>,
          class Key = typename Info::key_type,
          class T = typename Info::mapped_type,
          class Compare = std::less<Key>,
          class Allocator = std::allocator<std::pair<const Key, T>>>
    requires(!closure_allocator_like<Compare>)
map(std::from_range_t, R &&, Compare = Compare(), Allocator = Allocator())
    -> map<Key, T, Compare, Allocator>;

template <std::ranges::input_range R,
          class Info = closure_pair_info<std::ranges::range_value_t<R>>,
          class Key = typename Info::key_type,
          class T = typename Info::mapped_type,
          closure_allocator_like Allocator>
map(std::from_range_t, R &&, Allocator) -> map<Key, T, std::less<Key>, Allocator>;

template <std::input_iterator It,
          class Info = closure_pair_info<std::iter_value_t<It>>,
          class Key = typename Info::key_type,
          class T = typename Info::mapped_type,
          class Compare = std::less<Key>,
          class Allocator = std::allocator<std::pair<const Key, T>>>
    requires(!closure_allocator_like<Compare>)
multimap(It, It, Compare = Compare(), Allocator = Allocator())
    -> multimap<Key, T, Compare, Allocator>;

template <std::input_iterator It,
          class Info = closure_pair_info<std::iter_value_t<It>>,
          class Key = typename Info::key_type,
          class T = typename Info::mapped_type,
          closure_allocator_like Allocator>
multimap(It, It, Allocator) -> multimap<Key, T, std::less<Key>, Allocator>;

template <class Key, class T, class Compare = std::less<Key>,
          class Allocator = std::allocator<std::pair<const Key, T>>>
    requires(!closure_allocator_like<Compare>)
multimap(std::initializer_list<std::pair<Key, T>>, Compare = Compare(),
         Allocator = Allocator()) -> multimap<Key, T, Compare, Allocator>;

template <class Key, class T, closure_allocator_like Allocator>
multimap(std::initializer_list<std::pair<Key, T>>, Allocator)
    -> multimap<Key, T, std::less<Key>, Allocator>;

template <std::ranges::input_range R,
          class Info = closure_pair_info<std::ranges::range_value_t<R>>,
          class Key = typename Info::key_type,
          class T = typename Info::mapped_type,
          class Compare = std::less<Key>,
          class Allocator = std::allocator<std::pair<const Key, T>>>
    requires(!closure_allocator_like<Compare>)
multimap(std::from_range_t, R &&, Compare = Compare(),
         Allocator = Allocator()) -> multimap<Key, T, Compare, Allocator>;

template <std::ranges::input_range R,
          class Info = closure_pair_info<std::ranges::range_value_t<R>>,
          class Key = typename Info::key_type,
          class T = typename Info::mapped_type,
          closure_allocator_like Allocator>
multimap(std::from_range_t, R &&, Allocator)
    -> multimap<Key, T, std::less<Key>, Allocator>;

// --- ADL-friendly erasure (the standard also spells this as std::erase_if) --
template <class Key, class Compare, class Allocator, class Predicate>
std::size_t erase_if(set<Key, Compare, Allocator> &c, Predicate pred)
{
    std::size_t n = 0;
    for (auto it = c.begin(); it != c.end();)
    {
        if (pred(*it))
        {
            it = c.erase(it);
            ++n;
        }
        else
        {
            ++it;
        }
    }
    return n;
}

template <class Key, class Compare, class Allocator, class Predicate>
std::size_t erase_if(multiset<Key, Compare, Allocator> &c, Predicate pred)
{
    std::size_t n = 0;
    for (auto it = c.begin(); it != c.end();)
    {
        if (pred(*it))
        {
            it = c.erase(it);
            ++n;
        }
        else
        {
            ++it;
        }
    }
    return n;
}

template <class Key, class T, class Compare, class Allocator,
          class Predicate>
std::size_t erase_if(map<Key, T, Compare, Allocator> &c, Predicate pred)
{
    std::size_t n = 0;
    for (auto it = c.begin(); it != c.end();)
    {
        if (pred(*it))
        {
            it = c.erase(it);
            ++n;
        }
        else
        {
            ++it;
        }
    }
    return n;
}

template <class Key, class T, class Compare, class Allocator,
          class Predicate>
std::size_t erase_if(multimap<Key, T, Compare, Allocator> &c,
                     Predicate pred)
{
    std::size_t n = 0;
    for (auto it = c.begin(); it != c.end();)
    {
        if (pred(*it))
        {
            it = c.erase(it);
            ++n;
        }
        else
        {
            ++it;
        }
    }
    return n;
}

// ---------------------------------------------------------------------------
// Tests.
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Test support
// ---------------------------------------------------------------------------
namespace
{
    struct allocation_counter
    {
        static inline std::size_t live = 0;
    };

    template <class T>
    struct checking_allocator
    {
        using value_type = T;
        using is_always_equal = std::true_type;

        template <class U>
        struct rebind
        {
            using other = checking_allocator<U>;
        };

        checking_allocator() = default;
        template <class U>
        checking_allocator(const checking_allocator<U> &) noexcept {}

        T *allocate(std::size_t n)
        {
            ++allocation_counter::live;
            return static_cast<T *>(::operator new(n * sizeof(T)));
        }
        void deallocate(T *p, std::size_t) noexcept
        {
            --allocation_counter::live;
            ::operator delete(p);
        }

        template <class U>
        bool operator==(const checking_allocator<U> &) const noexcept
        {
            return true;
        }
    };

    struct throwing_less
    {
        static inline bool armed = false;
        bool operator()(int a, int b) const
        {
            if (armed)
            {
                throw std::runtime_error("throwing_less");
            }
            return a < b;
        }
    };

    struct unique_ptr_less
    {
        bool operator()(const std::unique_ptr<int> &a,
                        const std::unique_ptr<int> &b) const noexcept
        {
            return *a < *b;
        }
    };

    template <class C, class S>
    bool same_sequence(const C &c, const S &s)
    {
        if (c.size() != s.size())
        {
            return false;
        }
        auto i = c.begin();
        auto j = s.begin();
        for (; i != c.end(); ++i, ++j)
        {
            if (!(*i == *j))
            {
                return false;
            }
        }
        return j == s.end();
    }
} // namespace

static void test_differential()
{
    std::mt19937 rng(0xC0FFEEu);

    for (int round = 0; round < 40; ++round)
    {
        set<int> a;
        std::set<int> b;
        for (int step = 0; step < 300; ++step)
        {
            const int op = static_cast<int>(rng() % 10);
            const int x = static_cast<int>(rng() % 120);
            if (op < 5)
            {
                auto ra = a.insert(x);
                auto rb = b.insert(x);
                assert(ra.second == rb.second);
                assert(*ra.first == *rb.first);
            }
            else if (op < 7)
            {
                assert(a.erase(x) == b.erase(x));
            }
            else if (op < 8)
            {
                auto ia = a.find(x);
                auto ib = b.find(x);
                assert((ia == a.end()) == (ib == b.end()));
                if (ia != a.end())
                {
                    assert(*ia == *ib);
                }
            }
            else if (op < 9)
            {
                auto ia = a.lower_bound(x);
                auto ib = b.lower_bound(x);
                assert((ia == a.end()) == (ib == b.end()));
                if (ia != a.end())
                {
                    assert(*ia == *ib);
                }
                auto ua = a.upper_bound(x);
                auto ub = b.upper_bound(x);
                assert((ua == a.end()) == (ub == b.end()));
                if (ua != a.end())
                {
                    assert(*ua == *ub);
                }
            }
            else
            {
                assert(same_sequence(a, b));
            }
        }
        assert(same_sequence(a, b));

        auto ra = a.rbegin();
        auto rb = b.rbegin();
        while (ra != a.rend() && rb != b.rend())
        {
            assert(*ra == *rb);
            ++ra;
            ++rb;
        }
        assert(ra == a.rend());
        assert(rb == b.rend());
    }

    for (int round = 0; round < 25; ++round)
    {
        multiset<int> a;
        std::multiset<int> b;
        for (int step = 0; step < 300; ++step)
        {
            const int op = static_cast<int>(rng() % 10);
            const int x = static_cast<int>(rng() % 60);
            if (op < 6)
            {
                auto ia = a.insert(x);
                auto ib = b.insert(x);
                assert(*ia == *ib);
            }
            else if (op < 9)
            {
                assert(a.erase(x) == b.erase(x));
            }
            else
            {
                assert(same_sequence(a, b));
            }
        }
        assert(same_sequence(a, b));
    }

    for (int round = 0; round < 25; ++round)
    {
        map<int, int> a;
        std::map<int, int> b;
        for (int step = 0; step < 300; ++step)
        {
            const int op = static_cast<int>(rng() % 10);
            const int x = static_cast<int>(rng() % 80);
            const int y = static_cast<int>(rng() % 1000);
            if (op < 5)
            {
                auto ra = a.insert({x, y});
                auto rb = b.insert({x, y});
                assert(ra.second == rb.second);
                assert(ra.first->first == rb.first->first);
                assert(ra.first->second == rb.first->second);
            }
            else if (op < 7)
            {
                assert(a.erase(x) == b.erase(x));
            }
            else if (op < 8)
            {
                a[x] = y;
                b[x] = y;
            }
            else if (op < 9)
            {
                auto ia = a.find(x);
                auto ib = b.find(x);
                assert((ia == a.end()) == (ib == b.end()));
                if (ia != a.end())
                {
                    assert(ia->second == ib->second);
                }
            }
            else
            {
                assert(same_sequence(a, b));
            }
        }
        assert(same_sequence(a, b));
    }

    for (int round = 0; round < 20; ++round)
    {
        multimap<int, int> a;
        std::multimap<int, int> b;
        for (int step = 0; step < 200; ++step)
        {
            const int op = static_cast<int>(rng() % 10);
            const int x = static_cast<int>(rng() % 40);
            const int y = static_cast<int>(rng() % 1000);
            if (op < 6)
            {
                a.insert({x, y});
                b.insert({x, y});
            }
            else if (op < 9)
            {
                assert(a.erase(x) == b.erase(x));
            }
            else
            {
                assert(same_sequence(a, b));
            }
        }
        assert(same_sequence(a, b));
    }
}

static void test_ctad_and_ranges()
{
    set deduced_set{3, 1, 2, 2};
    multiset deduced_multiset{3, 1, 2, 2};
    map deduced_map{std::pair{1, 2}, std::pair{3, 4}};
    multimap deduced_multimap{std::pair{1, 2}, std::pair{1, 3}};

    static_assert(std::is_same_v<decltype(deduced_set), set<int>>);
    static_assert(std::is_same_v<decltype(deduced_multiset), multiset<int>>);
    static_assert(std::is_same_v<decltype(deduced_map), map<int, int>>);
    static_assert(
        std::is_same_v<decltype(deduced_multimap), multimap<int, int>>);

    assert(deduced_set.size() == 3);
    assert(deduced_multiset.size() == 4);
    assert(deduced_map.size() == 2);
    assert(deduced_multimap.size() == 2);

    std::vector<int> values{5, 4, 5};
    set from_iter(values.begin(), values.end());
    multiset ms_from_iter(values.begin(), values.end());
    set from_range(std::from_range, values);
    multiset ms_from_range(std::from_range, values);
    static_assert(std::is_same_v<decltype(from_iter), set<int>>);
    static_assert(std::is_same_v<decltype(ms_from_iter), multiset<int>>);
    static_assert(std::is_same_v<decltype(from_range), set<int>>);
    static_assert(std::is_same_v<decltype(ms_from_range), multiset<int>>);

    std::vector<std::pair<int, int>> pairs{{7, 8}, {9, 10}};
    map map_from_iter(pairs.begin(), pairs.end());
    multimap mm_from_iter(pairs.begin(), pairs.end());
    map map_from_range(std::from_range, pairs);
    multimap mm_from_range(std::from_range, pairs);
    static_assert(std::is_same_v<decltype(map_from_iter), map<int, int>>);
    static_assert(std::is_same_v<decltype(mm_from_iter), multimap<int, int>>);
    static_assert(std::is_same_v<decltype(map_from_range), map<int, int>>);
    static_assert(
        std::is_same_v<decltype(mm_from_range), multimap<int, int>>);

    std::vector<int> more{11, 12};
    from_range.insert_range(more);
    assert(from_range.contains(11) && from_range.contains(12));
}

static void test_move_only_and_node_handles()
{
    set<std::unique_ptr<int>, unique_ptr_less> s;
    s.insert(std::make_unique<int>(3));
    s.insert(std::make_unique<int>(1));
    s.insert(std::make_unique<int>(2));
    assert(s.size() == 3);
    assert(**s.begin() == 1);

    auto snh = s.extract(s.begin());
    assert(snh && snh.value() != nullptr);
    auto moved = std::move(snh.value());
    assert(*moved == 1);
    snh.value() = std::make_unique<int>(0);
    s.insert(std::move(snh));
    assert(!snh);
    assert(**s.begin() == 0);

    map<int, std::unique_ptr<int>> m;
    m.try_emplace(1, std::make_unique<int>(10));
    m.try_emplace(2, std::make_unique<int>(20));
    auto [it, inserted] = m.try_emplace(1, std::make_unique<int>(99));
    assert(!inserted);
    assert(*it->second == 10);

    auto nh = m.extract(1);
    assert(nh && nh.key() == 1 && *nh.mapped() == 10);
    nh.key() = 3;
    *nh.mapped() = 30;
    auto ret = m.insert(std::move(nh));
    assert(ret.inserted);
    assert(ret.position->first == 3);
    assert(*ret.position->second == 30);

    map<int, int> a{{1, 10}, {2, 20}, {3, 30}};
    map<int, int> b{{2, 200}, {4, 40}};
    b.merge(a);
    assert(b.size() == 4);
    assert(b.at(2) == 200);
    assert(a.size() == 1 && a.count(2) == 1);
    auto failed = b.insert(a.extract(2));
    assert(!failed.inserted && failed.node);
    b.insert(std::move(failed.node));
    assert(a.empty());
}

static void test_allocators_and_pmr()
{
    {
        set<int, std::less<int>, checking_allocator<int>> a;
        for (int i = 0; i < 200; ++i)
        {
            a.insert(i);
        }
        auto it = a.begin();
        std::advance(it, 100);
        auto b = a;
        assert(a == b);
        auto c = std::move(b);
        assert(*it == 100);
        c.erase(0);
        c.erase(199);
        c.clear();
        assert(c.empty());
    }
    assert(allocation_counter::live == 0);

    {
        set<int, throwing_less, checking_allocator<int>> s;
        for (int i = 0; i < 40; ++i)
        {
            s.insert(i);
        }
        const auto before = s.size();
        throwing_less::armed = true;
        bool threw = false;
        try
        {
            s.insert(1000);
        }
        catch (const std::runtime_error &)
        {
            threw = true;
        }
        throwing_less::armed = false;
        assert(threw);
        assert(s.size() == before);
        assert(allocation_counter::live == 40);
    }
    assert(allocation_counter::live == 0);

    {
        using ca = checking_allocator<int>;
        set<int, std::less<int>, ca> a;
        set<int, std::less<int>, ca> b;
        for (int i = 0; i < 20; ++i)
        {
            a.insert(i);
        }
        auto nh = a.extract(10);
        b.insert(std::move(nh));
        assert(!nh && a.size() == 19 && b.size() == 1);
        nh = a.extract(11);
        {
            auto discarded = std::move(nh);
            assert(discarded);
        }
        assert(!nh);
        assert(allocation_counter::live == 19);
    }
    assert(allocation_counter::live == 0);

    {
        std::pmr::monotonic_buffer_resource pool;
        using pa = std::pmr::polymorphic_allocator<std::pair<const int, int>>;
        map<int, int, std::less<int>, pa> m{pa(&pool)};
        for (int i = 0; i < 100; ++i)
        {
            m.try_emplace(i, i * i);
        }
        assert(m.get_allocator().resource() == &pool);
        assert(m.at(10) == 100);

        auto nh = m.extract(10);
        assert(nh.get_allocator().resource() == &pool);
        m.insert(std::move(nh));
        assert(m.at(10) == 100);

        std::pmr::monotonic_buffer_resource pool2;
        map<int, int, std::less<int>, pa> moved(std::move(m), pa(&pool2));
        assert(moved.get_allocator().resource() == &pool2);
        assert(moved.size() == 100);
        assert(m.empty());

        set<int, std::less<int>, std::pmr::polymorphic_allocator<int>> ps{
            std::pmr::polymorphic_allocator<int>(&pool2)};
        ps.insert(1);
        ps.insert(2);
        assert(ps.get_allocator().resource() == &pool2);
    }
}

static void test_iterators_swap_and_embedding()
{
    map<int, int> m;
    for (int i = 0; i < 100; ++i)
    {
        m.emplace(i, i);
    }
    auto it = m.find(50);
    const auto &cm = m;
    auto cit = cm.find(50);
    assert(it == cit);
    m.erase(1);
    m.insert({101, 101});
    m.erase(99);
    assert(it->first == 50 && it->second == 50);
    --it;
    assert(it->first == 49);
    auto last = m.end();
    --last;
    assert(last->first == 101);
    ++last;
    assert(last == m.end());

    map<int, int> other{{1000, 1000}};
    auto other_it = other.begin();
    swap(m, other);
    assert(other_it->first == 1000);
    assert(other.size() == 99);
    assert(m.size() == 1);

    erase_if(other, [](const auto &p)
             { return p.first % 2 == 0; });
    for (const auto &p : other)
    {
        assert(p.first % 2 != 0);
    }

    // The local engine type is a Voldemort type, yet it can be persisted as a
    // field of a local class thanks to decltype.
    auto local_factory = make_stl_toolkit;
    using local_core_t = decltype(local_factory.template operator()<int, int, std::identity,
                                                                    std::less<int>, false>(
        std::less<int>{}));
    struct holder
    {
        local_core_t engine;
    };
    holder h{local_factory.template operator()<int, int, std::identity,
                                               std::less<int>, false>(
        std::less<int>{})};
    // The engine is a tuple: index 1 is the element count and index 23 is
    // the reset closure.
    constexpr std::size_t engine_count = 1;
    constexpr std::size_t engine_reset = 23;
    std::get<engine_reset>(h.engine)(h.engine);
    assert(std::get<engine_count>(h.engine) == 0);
}

int main()
{
    test_differential();
    test_ctad_and_ranges();
    test_move_only_and_node_handles();
    test_allocators_and_pmr();
    test_iterators_swap_and_embedding();

    set<int> a{1, 2, 3};
    set<int> b{1, 2, 4};
    assert(a < b);
    assert(a != b);
    assert(a == a);

    assert(allocation_counter::live == 0);
    return 0;
}
