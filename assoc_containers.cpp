// A from-scratch, standard-conforming ordered associative container family.
//
// Architecture:
//   * Standard Wrapper + Message-Passing Closure Core
//   * The red-black tree is a *function*, not a struct of algorithms.  Calling
//     it with a message tag selects one compile-time branch; recursive
//     branches are given the function itself, in the style of a fixed point.
//   * Iterators are functions too: they answer value/next/previous messages,
//     and next/previous answer with the next iterator function.  This is the
//     coinductive-stream view of iteration.
//   * Node fields are never touched directly by algorithms: lens(node) returns
//     a message-dispatching closure, and every algorithm speaks through it.
//   * The public containers are ordinary global class templates; the tree
//     function's exact local type is obtained with decltype and stored as a
//     private member of their implementation base.  The wrappers and the
//     standard node-handle/iterator adapters are the only named types, because
//     the standard library requires them to be nameable and assignable.
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
#include <optional>
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
// outside the closure core; the engine itself contains no globals, and the only
// namespace-level declarations in the engine are message-tag aliases.
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
// Message tags.  They are ordinary empty values: a message is a type, and the
// receiver's generic operator() dispatches on that type with `if constexpr`.
// No algorithm is represented by a stored member or a named class.
// ---------------------------------------------------------------------------
namespace closure_msg
{
    // Node lens: reading and writing the physical cell through a closure.
    using left = std::integral_constant<int, 0>;
    using right = std::integral_constant<int, 1>;
    using parent = std::integral_constant<int, 2>;
    using red = std::integral_constant<int, 3>;
    using engaged = std::integral_constant<int, 4>;
    using value = std::integral_constant<int, 5>;
    using next = std::integral_constant<int, 6>;
    using prev = std::integral_constant<int, 7>;
    using set_left = std::integral_constant<int, 8>;
    using set_right = std::integral_constant<int, 9>;
    using set_parent = std::integral_constant<int, 10>;
    using set_red = std::integral_constant<int, 11>;
    using set_engaged = std::integral_constant<int, 12>;
    using set_next = std::integral_constant<int, 13>;
    using set_prev = std::integral_constant<int, 14>;

    // Tree messages.  Every one of them is a branch of the single tree
    // function; recursive branches receive the function itself as `self`.
    using tree_left_rotate = std::integral_constant<int, 15>;
    using tree_right_rotate = std::integral_constant<int, 16>;
    using tree_minimum = std::integral_constant<int, 17>;
    using tree_maximum = std::integral_constant<int, 18>;
    using tree_insert_fix = std::integral_constant<int, 19>;
    using tree_erase_fix = std::integral_constant<int, 20>;
    using tree_transplant = std::integral_constant<int, 21>;
    using tree_insert_node = std::integral_constant<int, 22>;
    using tree_insert_node_hint = std::integral_constant<int, 23>;
    using tree_detach = std::integral_constant<int, 24>;
    using tree_lower_bound = std::integral_constant<int, 25>;
    using tree_upper_bound = std::integral_constant<int, 26>;
    using tree_find_node = std::integral_constant<int, 27>;
    using tree_rebuild_threads = std::integral_constant<int, 28>;
    using tree_clear = std::integral_constant<int, 29>;
    using tree_clone = std::integral_constant<int, 30>;
    using tree_reset = std::integral_constant<int, 31>;
    using tree_adopt = std::integral_constant<int, 32>;
    using tree_take_from = std::integral_constant<int, 33>;
    using tree_swap_trees = std::integral_constant<int, 34>;
    using tree_first = std::integral_constant<int, 35>;
    using tree_last = std::integral_constant<int, 36>;
    using tree_root = std::integral_constant<int, 37>;
    using tree_end = std::integral_constant<int, 38>;
    using tree_next = std::integral_constant<int, 39>;
    using tree_prev = std::integral_constant<int, 40>;
    using tree_size = std::integral_constant<int, 41>;
    using tree_comp_ref = std::integral_constant<int, 42>;
    using tree_swap_comp = std::integral_constant<int, 43>;
    using tree_set_comp = std::integral_constant<int, 44>;
    using tree_link_header = std::integral_constant<int, 45>;
    using tree_header = std::integral_constant<int, 46>;
    using tree_make_iter = std::integral_constant<int, 47>;
    using tree_make_citer = std::integral_constant<int, 48>;

    // Iterator messages.  An iterator is a closure over one node; `iter_next`
    // and `iter_prev` answer with the next closure, exactly like a coinductive
    // stream whose tail is another stream.
    using iter_value = std::integral_constant<int, 49>;
    using iter_next = std::integral_constant<int, 50>;
    using iter_prev = std::integral_constant<int, 51>;
    using iter_equal = std::integral_constant<int, 52>;
    using iter_node = std::integral_constant<int, 53>;
    using iter_const = std::integral_constant<int, 54>;
} // namespace closure_msg

// ---------------------------------------------------------------------------
// The closure-core factory.  Every invocation produces a *new local type*.
// Five metadata parameters are the complete semantic configuration:
//   Key, Value, KeyOfValue, Compare, AllowDuplicates.
//
// What the factory returns is a single function.  It has private state (the
// header cell, the count, the comparator and the key extractor), and it
// evaluates one compile-time-selected branch for each message.  Recursive
// branches receive `self`, so the local type never has to be named inside
// itself: this is the Y-combinator shape, not a vtable or a struct of methods.
//
// The allocator is intentionally not part of the engine: the wrapper owns the
// allocator and supplies allocator-aware node lifetime operations.  This keeps
// the engine stateless with respect to memory resources and lets node_type
// carry the exact allocator required by the standard.
// ---------------------------------------------------------------------------
inline constexpr auto make_stl_toolkit =
    []<typename Key, typename Value, typename KeyOfValue, typename Compare,
       bool AllowDuplicates>(Compare comp = Compare{})
{
    // The physical cell.  This is the only aggregate in the engine: it is the
    // storage the allocator must be able to name and rebind, not a semantic
    // object.  Algorithms never touch its fields directly; they send messages
    // through the lens closure below.
    struct node
    {
        node *parent = nullptr;
        node *left = nullptr;
        node *right = nullptr;
        node *next = nullptr;
        node *prev = nullptr;
        bool red = false;
        bool engaged = false;
        union
        {
            Value value;
        };

        node() noexcept
            : parent(nullptr),
              left(nullptr),
              right(nullptr),
              next(nullptr),
              prev(nullptr),
              red(false),
              engaged(false) {}

        // Only the header is ever moved: its value member is not alive.
        // Copying a live tree would alias its nodes, so the tree function is
        // deliberately move-only.  The wrapper implements standard copy
        // semantics by cloning through the tree_clone message.
        node(const node &) = delete;
        node &operator=(const node &) = delete;

        node(node &&other) noexcept
            : parent(other.parent),
              left(other.left),
              right(other.right),
              next(other.next),
              prev(other.prev),
              red(other.red),
              engaged(other.engaged)
        {
            // The header has just changed address.  Redirect the root's
            // parent link and the two thread endpoints at the new header.
            if (!other.engaged)
            {
                if (left)
                {
                    left->parent = this;
                }
                if (next == &other)
                {
                    next = this;
                }
                else if (next)
                {
                    next->prev = this;
                }
                if (prev == &other)
                {
                    prev = this;
                }
                else if (prev)
                {
                    prev->next = this;
                }
                parent = left;
            }
        }

        node &operator=(node &&) = delete;

        ~node() noexcept {}

        Value *closure_value() noexcept { return std::addressof(value); }
        bool closure_engaged() const noexcept { return engaged; }
        void closure_set_engaged(bool v) noexcept { engaged = v; }
    };

    // -------------------------------------------------------------------
    // The lens.  `lens(node)` returns a closure that answers field messages.
    // This is the only code in the engine that mentions a node member.
    // -------------------------------------------------------------------
    auto lens = [](node *p) noexcept
    {
        return [p]<class Tag, class... Args>(Tag,
                                             Args &&...args) noexcept
                   -> decltype(auto)
        {
            using T = std::remove_cvref_t<Tag>;
            if constexpr (std::same_as<T, closure_msg::left>)
            {
                return static_cast<node *const &>(p->left);
            }
            else if constexpr (std::same_as<T, closure_msg::right>)
            {
                return static_cast<node *const &>(p->right);
            }
            else if constexpr (std::same_as<T, closure_msg::parent>)
            {
                return static_cast<node *const &>(p->parent);
            }
            else if constexpr (std::same_as<T, closure_msg::red>)
            {
                return static_cast<bool const &>(p->red);
            }
            else if constexpr (std::same_as<T, closure_msg::engaged>)
            {
                return static_cast<bool const &>(p->engaged);
            }
            else if constexpr (std::same_as<T, closure_msg::value>)
            {
                return (p->value);
            }
            else if constexpr (std::same_as<T, closure_msg::next>)
            {
                return static_cast<node *const &>(p->next);
            }
            else if constexpr (std::same_as<T, closure_msg::prev>)
            {
                return static_cast<node *const &>(p->prev);
            }
            else if constexpr (std::same_as<T, closure_msg::set_left>)
            {
                static_assert(sizeof...(Args) == 1);
                p->left = (std::forward<Args>(args), ...);
            }
            else if constexpr (std::same_as<T, closure_msg::set_right>)
            {
                static_assert(sizeof...(Args) == 1);
                p->right = (std::forward<Args>(args), ...);
            }
            else if constexpr (std::same_as<T, closure_msg::set_parent>)
            {
                static_assert(sizeof...(Args) == 1);
                p->parent = (std::forward<Args>(args), ...);
            }
            else if constexpr (std::same_as<T, closure_msg::set_red>)
            {
                static_assert(sizeof...(Args) == 1);
                p->red = (std::forward<Args>(args), ...);
            }
            else if constexpr (std::same_as<T, closure_msg::set_engaged>)
            {
                static_assert(sizeof...(Args) == 1);
                p->engaged = (std::forward<Args>(args), ...);
            }
            else if constexpr (std::same_as<T, closure_msg::set_next>)
            {
                static_assert(sizeof...(Args) == 1);
                p->next = (std::forward<Args>(args), ...);
            }
            else if constexpr (std::same_as<T, closure_msg::set_prev>)
            {
                static_assert(sizeof...(Args) == 1);
                p->prev = (std::forward<Args>(args), ...);
            }
            else
            {
                static_assert(!sizeof(Tag), "unknown node-lens message");
            }
        };
    };

    // -------------------------------------------------------------------
    // Coinductive iterators.  There is no iterator struct here: an iterator
    // is a function from messages to answers.  `iter_next` and `iter_prev`
    // answer with the next iterator function, so advancing an iterator is
    // applying it, not mutating a cursor field.
    // -------------------------------------------------------------------
    auto make_citer = [lens](auto self, node *p)
    {
        return [self, lens, p](auto tag, auto &&...args) -> decltype(auto)
        {
            using T = std::remove_cvref_t<decltype(tag)>;
            if constexpr (std::same_as<T, closure_msg::iter_value>)
            {
                return std::as_const(lens(p)(closure_msg::value{}));
            }
            else if constexpr (std::same_as<T, closure_msg::iter_next>)
            {
                return self(self, lens(p)(closure_msg::next{}));
            }
            else if constexpr (std::same_as<T, closure_msg::iter_prev>)
            {
                return self(self, lens(p)(closure_msg::prev{}));
            }
            else if constexpr (std::same_as<T, closure_msg::iter_equal>)
            {
                return p == (std::forward<decltype(args)>(args)(
                                 closure_msg::iter_node{}),
                             ...);
            }
            else if constexpr (std::same_as<T, closure_msg::iter_node>)
            {
                return p;
            }
            else
            {
                static_assert(!sizeof(T), "unknown iterator message");
            }
        };
    };

    auto make_iter = [lens](auto self, auto cself, node *p)
    {
        return [self, cself, lens, p](auto tag,
                                      auto &&...args) -> decltype(auto)
        {
            using T = std::remove_cvref_t<decltype(tag)>;
            if constexpr (std::same_as<T, closure_msg::iter_value>)
            {
                return lens(p)(closure_msg::value{});
            }
            else if constexpr (std::same_as<T, closure_msg::iter_next>)
            {
                return self(self, cself, lens(p)(closure_msg::next{}));
            }
            else if constexpr (std::same_as<T, closure_msg::iter_prev>)
            {
                return self(self, cself, lens(p)(closure_msg::prev{}));
            }
            else if constexpr (std::same_as<T, closure_msg::iter_equal>)
            {
                return p == (std::forward<decltype(args)>(args)(
                                 closure_msg::iter_node{}),
                             ...);
            }
            else if constexpr (std::same_as<T, closure_msg::iter_node>)
            {
                return p;
            }
            else if constexpr (std::same_as<T, closure_msg::iter_const>)
            {
                return cself(cself, p);
            }
            else
            {
                static_assert(!sizeof(T), "unknown iterator message");
            }
        };
    };

    // -------------------------------------------------------------------
    // The tree.  One function, one dispatcher, no stored operation members.
    // -------------------------------------------------------------------
    auto tree = [header = node{}, count = std::size_t{0},
                 comp = std::move(comp), key_of = KeyOfValue{},
                 lens, make_iter, make_citer](auto &self, auto tag,
                                              auto &&...args) mutable
        -> decltype(auto)
    {
        using T = std::remove_cvref_t<decltype(tag)>;

        // --- rotations -------------------------------------------------
        if constexpr (std::same_as<T, closure_msg::tree_left_rotate>)
        {
            node *x = std::get<0>(std::forward_as_tuple(args...));
            node *y = lens(x)(closure_msg::right{});
            lens(x)(closure_msg::set_right{},
                    lens(y)(closure_msg::left{}));
            if (lens(y)(closure_msg::left{}))
            {
                lens(lens(y)(closure_msg::left{}))(
                    closure_msg::set_parent{}, x);
            }
            node *xp = lens(x)(closure_msg::parent{});
            lens(y)(closure_msg::set_parent{}, xp);
            if (xp == &header)
            {
                lens(&header)(closure_msg::set_left{}, y);
            }
            else if (x == lens(xp)(closure_msg::left{}))
            {
                lens(xp)(closure_msg::set_left{}, y);
            }
            else
            {
                lens(xp)(closure_msg::set_right{}, y);
            }
            lens(y)(closure_msg::set_left{}, x);
            lens(x)(closure_msg::set_parent{}, y);
        }
        else if constexpr (std::same_as<T, closure_msg::tree_right_rotate>)
        {
            node *x = std::get<0>(std::forward_as_tuple(args...));
            node *y = lens(x)(closure_msg::left{});
            lens(x)(closure_msg::set_left{},
                    lens(y)(closure_msg::right{}));
            if (lens(y)(closure_msg::right{}))
            {
                lens(lens(y)(closure_msg::right{}))(
                    closure_msg::set_parent{}, x);
            }
            node *xp = lens(x)(closure_msg::parent{});
            lens(y)(closure_msg::set_parent{}, xp);
            if (xp == &header)
            {
                lens(&header)(closure_msg::set_left{}, y);
            }
            else if (x == lens(xp)(closure_msg::right{}))
            {
                lens(xp)(closure_msg::set_right{}, y);
            }
            else
            {
                lens(xp)(closure_msg::set_left{}, y);
            }
            lens(y)(closure_msg::set_right{}, x);
            lens(x)(closure_msg::set_parent{}, y);
        }
        // --- structural search ----------------------------------------
        else if constexpr (std::same_as<T, closure_msg::tree_minimum>)
        {
            node *x = std::get<0>(std::forward_as_tuple(args...));
            while (x && lens(x)(closure_msg::left{}))
            {
                x = lens(x)(closure_msg::left{});
            }
            return x;
        }
        else if constexpr (std::same_as<T, closure_msg::tree_maximum>)
        {
            node *x = std::get<0>(std::forward_as_tuple(args...));
            while (x && lens(x)(closure_msg::right{}))
            {
                x = lens(x)(closure_msg::right{});
            }
            return x;
        }
        // --- insertion fix-up (CLRS) -----------------------------------
        else if constexpr (std::same_as<T, closure_msg::tree_insert_fix>)
        {
            node *z = std::get<0>(std::forward_as_tuple(args...));
            auto root = [&]() noexcept
            {
                return lens(&header)(closure_msg::left{});
            };
            auto is_red = [&](node *n) noexcept
            {
                return n != nullptr && lens(n)(closure_msg::red{});
            };
            while (z != root() && is_red(lens(z)(closure_msg::parent{})))
            {
                node *p = lens(z)(closure_msg::parent{});
                node *g = lens(p)(closure_msg::parent{});
                if (p == lens(g)(closure_msg::left{}))
                {
                    node *u = lens(g)(closure_msg::right{});
                    if (is_red(u))
                    {
                        lens(p)(closure_msg::set_red{}, false);
                        lens(u)(closure_msg::set_red{}, false);
                        lens(g)(closure_msg::set_red{}, true);
                        z = g;
                    }
                    else
                    {
                        if (z == lens(p)(closure_msg::right{}))
                        {
                            z = p;
                            self(self, closure_msg::tree_left_rotate{}, z);
                            p = lens(z)(closure_msg::parent{});
                            g = lens(p)(closure_msg::parent{});
                        }
                        lens(p)(closure_msg::set_red{}, false);
                        lens(g)(closure_msg::set_red{}, true);
                        self(self, closure_msg::tree_right_rotate{}, g);
                    }
                }
                else
                {
                    node *u = lens(g)(closure_msg::left{});
                    if (is_red(u))
                    {
                        lens(p)(closure_msg::set_red{}, false);
                        lens(u)(closure_msg::set_red{}, false);
                        lens(g)(closure_msg::set_red{}, true);
                        z = g;
                    }
                    else
                    {
                        if (z == lens(p)(closure_msg::left{}))
                        {
                            z = p;
                            self(self, closure_msg::tree_right_rotate{}, z);
                            p = lens(z)(closure_msg::parent{});
                            g = lens(p)(closure_msg::parent{});
                        }
                        lens(p)(closure_msg::set_red{}, false);
                        lens(g)(closure_msg::set_red{}, true);
                        self(self, closure_msg::tree_left_rotate{}, g);
                    }
                }
            }
            if (root())
            {
                lens(root())(closure_msg::set_red{}, false);
            }
        }
        // --- deletion fix-up -------------------------------------------
        else if constexpr (std::same_as<T, closure_msg::tree_erase_fix>)
        {
            auto &&x = std::get<0>(std::forward_as_tuple(args...));
            auto &&xp = std::get<1>(std::forward_as_tuple(args...));
            auto is_red = [&](node *n) noexcept
            {
                return n != nullptr && lens(n)(closure_msg::red{});
            };
            while (x != lens(&header)(closure_msg::left{}) && !is_red(x))
            {
                if (xp == nullptr)
                {
                    break;
                }
                if (x == lens(xp)(closure_msg::left{}))
                {
                    node *w = lens(xp)(closure_msg::right{});
                    if (w == nullptr)
                    {
                        x = xp;
                        xp = lens(x)(closure_msg::parent{});
                        continue;
                    }
                    if (is_red(w))
                    {
                        lens(w)(closure_msg::set_red{}, false);
                        lens(xp)(closure_msg::set_red{}, true);
                        self(self, closure_msg::tree_left_rotate{}, xp);
                        w = lens(xp)(closure_msg::right{});
                    }
                    if (!is_red(lens(w)(closure_msg::left{})) &&
                        !is_red(lens(w)(closure_msg::right{})))
                    {
                        lens(w)(closure_msg::set_red{}, true);
                        x = xp;
                        xp = lens(x)(closure_msg::parent{});
                    }
                    else
                    {
                        if (!is_red(lens(w)(closure_msg::right{})))
                        {
                            if (lens(w)(closure_msg::left{}))
                            {
                                lens(lens(w)(closure_msg::left{}))(
                                    closure_msg::set_red{}, false);
                            }
                            lens(w)(closure_msg::set_red{}, true);
                            self(self, closure_msg::tree_right_rotate{}, w);
                            w = lens(xp)(closure_msg::right{});
                        }
                        lens(w)(closure_msg::set_red{},
                                lens(xp)(closure_msg::red{}));
                        lens(xp)(closure_msg::set_red{}, false);
                        if (lens(w)(closure_msg::right{}))
                        {
                            lens(lens(w)(closure_msg::right{}))(
                                closure_msg::set_red{}, false);
                        }
                        self(self, closure_msg::tree_left_rotate{}, xp);
                        x = lens(&header)(closure_msg::left{});
                        xp = nullptr;
                    }
                }
                else
                {
                    node *w = lens(xp)(closure_msg::left{});
                    if (w == nullptr)
                    {
                        x = xp;
                        xp = lens(x)(closure_msg::parent{});
                        continue;
                    }
                    if (is_red(w))
                    {
                        lens(w)(closure_msg::set_red{}, false);
                        lens(xp)(closure_msg::set_red{}, true);
                        self(self, closure_msg::tree_right_rotate{}, xp);
                        w = lens(xp)(closure_msg::left{});
                    }
                    if (!is_red(lens(w)(closure_msg::right{})) &&
                        !is_red(lens(w)(closure_msg::left{})))
                    {
                        lens(w)(closure_msg::set_red{}, true);
                        x = xp;
                        xp = lens(x)(closure_msg::parent{});
                    }
                    else
                    {
                        if (!is_red(lens(w)(closure_msg::left{})))
                        {
                            if (lens(w)(closure_msg::right{}))
                            {
                                lens(lens(w)(closure_msg::right{}))(
                                    closure_msg::set_red{}, false);
                            }
                            lens(w)(closure_msg::set_red{}, true);
                            self(self, closure_msg::tree_left_rotate{}, w);
                            w = lens(xp)(closure_msg::left{});
                        }
                        lens(w)(closure_msg::set_red{},
                                lens(xp)(closure_msg::red{}));
                        lens(xp)(closure_msg::set_red{}, false);
                        if (lens(w)(closure_msg::left{}))
                        {
                            lens(lens(w)(closure_msg::left{}))(
                                closure_msg::set_red{}, false);
                        }
                        self(self, closure_msg::tree_right_rotate{}, xp);
                        x = lens(&header)(closure_msg::left{});
                        xp = nullptr;
                    }
                }
            }
            if (x)
            {
                lens(x)(closure_msg::set_red{}, false);
            }
        }
        else if constexpr (std::same_as<T, closure_msg::tree_transplant>)
        {
            node *u = std::get<0>(std::forward_as_tuple(args...));
            node *v = std::get<1>(std::forward_as_tuple(args...));
            node *up = lens(u)(closure_msg::parent{});
            if (up == &header)
            {
                lens(&header)(closure_msg::set_left{}, v);
            }
            else if (u == lens(up)(closure_msg::left{}))
            {
                lens(up)(closure_msg::set_left{}, v);
            }
            else
            {
                lens(up)(closure_msg::set_right{}, v);
            }
            if (v)
            {
                lens(v)(closure_msg::set_parent{}, up);
            }
        }
        // --- insertion -------------------------------------------------
        else if constexpr (std::same_as<T, closure_msg::tree_insert_node>)
        {
            node *z = std::get<0>(std::forward_as_tuple(args...));
            node *y = &header;
            node *x = lens(&header)(closure_msg::left{});
            bool went_left = false;
            while (x)
            {
                y = x;
                const Key &zk =
                    key_of(lens(z)(closure_msg::value{}));
                const Key &xk =
                    key_of(lens(x)(closure_msg::value{}));
                const bool before = comp(zk, xk);
                if constexpr (!AllowDuplicates)
                {
                    const bool after = comp(xk, zk);
                    if (!before && !after)
                    {
                        return std::pair<node *, bool>{x, false};
                    }
                }
                if (before)
                {
                    x = lens(x)(closure_msg::left{});
                    went_left = true;
                }
                else
                {
                    x = lens(x)(closure_msg::right{});
                    went_left = false;
                }
            }
            lens(z)(closure_msg::set_parent{}, y);
            lens(z)(closure_msg::set_left{},
                    static_cast<node *>(nullptr));
            lens(z)(closure_msg::set_right{},
                    static_cast<node *>(nullptr));
            lens(z)(closure_msg::set_red{}, true);
            if (y == &header)
            {
                lens(&header)(closure_msg::set_left{}, z);
                lens(&header)(closure_msg::set_parent{}, z);
                lens(z)(closure_msg::set_next{}, &header);
                lens(z)(closure_msg::set_prev{}, &header);
                lens(&header)(closure_msg::set_next{}, z);
                lens(&header)(closure_msg::set_prev{}, z);
            }
            else if (went_left)
            {
                lens(y)(closure_msg::set_left{}, z);
                node *pv = lens(y)(closure_msg::prev{});
                lens(z)(closure_msg::set_next{}, y);
                lens(z)(closure_msg::set_prev{}, pv);
                lens(pv)(closure_msg::set_next{}, z);
                lens(y)(closure_msg::set_prev{}, z);
            }
            else
            {
                lens(y)(closure_msg::set_right{}, z);
                node *nx = lens(y)(closure_msg::next{});
                lens(z)(closure_msg::set_prev{}, y);
                lens(z)(closure_msg::set_next{}, nx);
                lens(nx)(closure_msg::set_prev{}, z);
                lens(y)(closure_msg::set_next{}, z);
            }
            ++count;
            self(self, closure_msg::tree_insert_fix{}, z);
            node *r = lens(&header)(closure_msg::left{});
            lens(&header)(closure_msg::set_parent{}, r);
            return std::pair<node *, bool>{z, true};
        }
        else if constexpr (std::same_as<T,
                                        closure_msg::tree_insert_node_hint>)
        {
            node *z = std::get<0>(std::forward_as_tuple(args...));
            node *hint = std::get<1>(std::forward_as_tuple(args...));
            node *root = lens(&header)(closure_msg::left{});
            if (!root)
            {
                return self(self, closure_msg::tree_insert_node{}, z);
            }
            const Key &zk = key_of(lens(z)(closure_msg::value{}));
            if (hint == &header)
            {
                node *last = lens(&header)(closure_msg::prev{});
                const Key &lk = key_of(lens(last)(closure_msg::value{}));
                const bool valid =
                    AllowDuplicates ? !comp(zk, lk) : comp(lk, zk);
                if (!valid)
                {
                    return self(self, closure_msg::tree_insert_node{}, z);
                }
                lens(z)(closure_msg::set_parent{}, last);
                lens(z)(closure_msg::set_left{},
                        static_cast<node *>(nullptr));
                lens(z)(closure_msg::set_right{},
                        static_cast<node *>(nullptr));
                lens(z)(closure_msg::set_red{}, true);
                lens(last)(closure_msg::set_right{}, z);
                lens(z)(closure_msg::set_prev{}, last);
                lens(z)(closure_msg::set_next{}, &header);
                lens(last)(closure_msg::set_next{}, z);
                lens(&header)(closure_msg::set_prev{}, z);
                ++count;
                self(self, closure_msg::tree_insert_fix{}, z);
                lens(&header)(closure_msg::set_parent{},
                              lens(&header)(closure_msg::left{}));
                return std::pair<node *, bool>{z, true};
            }
            const Key &hk = key_of(lens(hint)(closure_msg::value{}));
            if constexpr (AllowDuplicates)
            {
                if (comp(hk, zk))
                {
                    return self(self, closure_msg::tree_insert_node{}, z);
                }
            }
            else
            {
                if (!comp(zk, hk))
                {
                    return self(self, closure_msg::tree_insert_node{}, z);
                }
            }
            node *pred = lens(hint)(closure_msg::prev{});
            if (pred != &header)
            {
                const Key &pk =
                    key_of(lens(pred)(closure_msg::value{}));
                if constexpr (AllowDuplicates)
                {
                    if (comp(zk, pk))
                    {
                        return self(self, closure_msg::tree_insert_node{},
                                    z);
                    }
                }
                else
                {
                    if (!comp(pk, zk))
                    {
                        return self(self, closure_msg::tree_insert_node{},
                                    z);
                    }
                }
            }
            node *parent = nullptr;
            if (lens(hint)(closure_msg::left{}) == nullptr)
            {
                parent = hint;
                lens(hint)(closure_msg::set_left{}, z);
            }
            else
            {
                parent = self(self, closure_msg::tree_maximum{},
                              lens(hint)(closure_msg::left{}));
                lens(parent)(closure_msg::set_right{}, z);
            }
            lens(z)(closure_msg::set_parent{}, parent);
            lens(z)(closure_msg::set_left{},
                    static_cast<node *>(nullptr));
            lens(z)(closure_msg::set_right{},
                    static_cast<node *>(nullptr));
            lens(z)(closure_msg::set_red{}, true);
            lens(z)(closure_msg::set_prev{}, pred);
            lens(z)(closure_msg::set_next{}, hint);
            lens(pred)(closure_msg::set_next{}, z);
            lens(hint)(closure_msg::set_prev{}, z);
            ++count;
            self(self, closure_msg::tree_insert_fix{}, z);
            lens(&header)(closure_msg::set_parent{},
                          lens(&header)(closure_msg::left{}));
            return std::pair<node *, bool>{z, true};
        }
        // --- deletion --------------------------------------------------
        else if constexpr (std::same_as<T, closure_msg::tree_detach>)
        {
            node *z = std::get<0>(std::forward_as_tuple(args...));
            node *zn = lens(z)(closure_msg::next{});
            node *zp = lens(z)(closure_msg::prev{});
            lens(zp)(closure_msg::set_next{}, zn);
            lens(zn)(closure_msg::set_prev{}, zp);
            node *y = z;
            bool y_was_red = lens(y)(closure_msg::red{});
            node *x = nullptr;
            node *xp = nullptr;
            if (lens(z)(closure_msg::left{}) == nullptr)
            {
                x = lens(z)(closure_msg::right{});
                xp = lens(z)(closure_msg::parent{});
                self(self, closure_msg::tree_transplant{}, z, x);
            }
            else if (lens(z)(closure_msg::right{}) == nullptr)
            {
                x = lens(z)(closure_msg::left{});
                xp = lens(z)(closure_msg::parent{});
                self(self, closure_msg::tree_transplant{}, z, x);
            }
            else
            {
                y = self(self, closure_msg::tree_minimum{},
                         lens(z)(closure_msg::right{}));
                y_was_red = lens(y)(closure_msg::red{});
                x = lens(y)(closure_msg::right{});
                if (lens(y)(closure_msg::parent{}) == z)
                {
                    xp = y;
                    if (x)
                    {
                        lens(x)(closure_msg::set_parent{}, y);
                    }
                }
                else
                {
                    xp = lens(y)(closure_msg::parent{});
                    self(self, closure_msg::tree_transplant{}, y, x);
                    lens(y)(closure_msg::set_right{},
                            lens(z)(closure_msg::right{}));
                    lens(lens(y)(closure_msg::right{}))(
                        closure_msg::set_parent{}, y);
                }
                self(self, closure_msg::tree_transplant{}, z, y);
                lens(y)(closure_msg::set_left{},
                        lens(z)(closure_msg::left{}));
                lens(lens(y)(closure_msg::left{}))(
                    closure_msg::set_parent{}, y);
                lens(y)(closure_msg::set_red{},
                        lens(z)(closure_msg::red{}));
            }
            --count;
            if (!y_was_red)
            {
                self(self, closure_msg::tree_erase_fix{}, x, xp);
            }
            node *r = lens(&header)(closure_msg::left{});
            if (r)
            {
                lens(r)(closure_msg::set_parent{}, &header);
            }
            lens(&header)(closure_msg::set_parent{}, r);
            lens(z)(closure_msg::set_left{},
                    static_cast<node *>(nullptr));
            lens(z)(closure_msg::set_right{},
                    static_cast<node *>(nullptr));
            lens(z)(closure_msg::set_parent{},
                    static_cast<node *>(nullptr));
            lens(z)(closure_msg::set_next{},
                    static_cast<node *>(nullptr));
            lens(z)(closure_msg::set_prev{},
                    static_cast<node *>(nullptr));
            lens(z)(closure_msg::set_red{}, false);
        }
        // --- lookup ----------------------------------------------------
        else if constexpr (std::same_as<T, closure_msg::tree_lower_bound>)
        {
            auto &&key = std::get<0>(std::forward_as_tuple(args...));
            node *x = lens(&header)(closure_msg::left{});
            node *y = &header;
            while (x)
            {
                const Key &xk =
                    key_of(lens(x)(closure_msg::value{}));
                if (!comp(xk, key))
                {
                    y = x;
                    x = lens(x)(closure_msg::left{});
                }
                else
                {
                    x = lens(x)(closure_msg::right{});
                }
            }
            return y;
        }
        else if constexpr (std::same_as<T, closure_msg::tree_upper_bound>)
        {
            auto &&key = std::get<0>(std::forward_as_tuple(args...));
            node *x = lens(&header)(closure_msg::left{});
            node *y = &header;
            while (x)
            {
                const Key &xk =
                    key_of(lens(x)(closure_msg::value{}));
                if (comp(key, xk))
                {
                    y = x;
                    x = lens(x)(closure_msg::left{});
                }
                else
                {
                    x = lens(x)(closure_msg::right{});
                }
            }
            return y;
        }
        else if constexpr (std::same_as<T, closure_msg::tree_find_node>)
        {
            auto &&key = std::get<0>(std::forward_as_tuple(args...));
            node *n = self(self, closure_msg::tree_lower_bound{}, key);
            if (n != &header)
            {
                const Key &nk =
                    key_of(lens(n)(closure_msg::value{}));
                if (!comp(key, nk) && !comp(nk, key))
                {
                    return n;
                }
            }
            return static_cast<node *>(&header);
        }
        // --- structural utilities --------------------------------------
        else if constexpr (std::same_as<T,
                                        closure_msg::tree_rebuild_threads>)
        {
            node *r = lens(&header)(closure_msg::left{});
            node *prev = &header;
            auto visit = [&](auto &&go, node *n) -> void
            {
                if (!n)
                {
                    return;
                }
                go(go, lens(n)(closure_msg::left{}));
                lens(prev)(closure_msg::set_next{}, n);
                lens(n)(closure_msg::set_prev{}, prev);
                prev = n;
                go(go, lens(n)(closure_msg::right{}));
            };
            if (r)
            {
                visit(visit, r);
            }
            lens(prev)(closure_msg::set_next{}, &header);
            lens(&header)(closure_msg::set_prev{}, prev);
            lens(&header)(closure_msg::set_next{},
                          r ? self(self, closure_msg::tree_minimum{}, r)
                            : &header);
            lens(&header)(closure_msg::set_parent{}, r);
        }
        else if constexpr (std::same_as<T, closure_msg::tree_clear>)
        {
            node *n = std::get<0>(std::forward_as_tuple(args...));
            auto &&destroy =
                std::get<1>(std::forward_as_tuple(args...));
            if (!n)
            {
                return;
            }
            node *l = lens(n)(closure_msg::left{});
            node *r = lens(n)(closure_msg::right{});
            self(self, closure_msg::tree_clear{}, l, destroy);
            self(self, closure_msg::tree_clear{}, r, destroy);
            destroy(n);
        }
        else if constexpr (std::same_as<T, closure_msg::tree_clone>)
        {
            node *src = std::get<0>(std::forward_as_tuple(args...));
            auto &&create = std::get<1>(std::forward_as_tuple(args...));
            auto &&destroy =
                std::get<2>(std::forward_as_tuple(args...));
            if (!src)
            {
                return static_cast<node *>(nullptr);
            }
            node *d = create(src);
            try
            {
                node *l = self(self, closure_msg::tree_clone{},
                               lens(src)(closure_msg::left{}), create,
                               destroy);
                lens(d)(closure_msg::set_left{}, l);
                if (l)
                {
                    lens(l)(closure_msg::set_parent{}, d);
                }
                node *r = self(self, closure_msg::tree_clone{},
                               lens(src)(closure_msg::right{}), create,
                               destroy);
                lens(d)(closure_msg::set_right{}, r);
                if (r)
                {
                    lens(r)(closure_msg::set_parent{}, d);
                }
                lens(d)(closure_msg::set_red{},
                        lens(src)(closure_msg::red{}));
                return d;
            }
            catch (...)
            {
                self(self, closure_msg::tree_clear{}, d, destroy);
                throw;
            }
        }
        else if constexpr (std::same_as<T, closure_msg::tree_reset>)
        {
            lens(&header)(closure_msg::set_left{},
                          static_cast<node *>(nullptr));
            lens(&header)(closure_msg::set_right{},
                          static_cast<node *>(nullptr));
            lens(&header)(closure_msg::set_parent{},
                          static_cast<node *>(nullptr));
            lens(&header)(closure_msg::set_next{}, &header);
            lens(&header)(closure_msg::set_prev{}, &header);
            count = 0;
        }
        else if constexpr (std::same_as<T, closure_msg::tree_adopt>)
        {
            node *r = std::get<0>(std::forward_as_tuple(args...));
            std::size_t n = std::get<1>(std::forward_as_tuple(args...));
            lens(&header)(closure_msg::set_left{}, r);
            if (r)
            {
                lens(r)(closure_msg::set_parent{}, &header);
            }
            lens(&header)(closure_msg::set_parent{}, r);
            count = n;
            self(self, closure_msg::tree_rebuild_threads{});
        }
        // Link a pre-existing tree, whose thread endpoints currently point at
        // `foreign_header`, into this tree's header.  This is the primitive
        // behind take_from and swap_trees.
        else if constexpr (std::same_as<T,
                                        closure_msg::tree_link_header>)
        {
            node *root = std::get<0>(std::forward_as_tuple(args...));
            node *first = std::get<1>(std::forward_as_tuple(args...));
            node *last = std::get<2>(std::forward_as_tuple(args...));
            std::size_t n = std::get<3>(std::forward_as_tuple(args...));
            node *foreign_header =
                std::get<4>(std::forward_as_tuple(args...));
            lens(&header)(closure_msg::set_left{}, root);
            lens(&header)(closure_msg::set_parent{}, root);
            lens(&header)(closure_msg::set_next{},
                          first == foreign_header ? &header : first);
            lens(&header)(closure_msg::set_prev{},
                          last == foreign_header ? &header : last);
            count = n;
            if (root)
            {
                lens(root)(closure_msg::set_parent{}, &header);
                if (first != foreign_header)
                {
                    lens(first)(closure_msg::set_prev{}, &header);
                    lens(last)(closure_msg::set_next{}, &header);
                }
            }
        }
        else if constexpr (std::same_as<T, closure_msg::tree_take_from>)
        {
            auto &src = std::get<0>(std::forward_as_tuple(args...));
            node *src_header = src(src, closure_msg::tree_header{});
            node *r = src(src, closure_msg::tree_root{});
            node *first = src(src, closure_msg::tree_first{});
            node *last = src(src, closure_msg::tree_last{});
            std::size_t n = src(src, closure_msg::tree_size{});
            self(self, closure_msg::tree_link_header{}, r, first, last, n,
                 src_header);
            src(src, closure_msg::tree_reset{});
        }
        else if constexpr (std::same_as<T, closure_msg::tree_swap_trees>)
        {
            auto &other = std::get<0>(std::forward_as_tuple(args...));
            node *ah = self(self, closure_msg::tree_header{});
            node *bh = other(other, closure_msg::tree_header{});
            node *ar = self(self, closure_msg::tree_root{});
            node *br = other(other, closure_msg::tree_root{});
            node *af = self(self, closure_msg::tree_first{});
            node *al = self(self, closure_msg::tree_last{});
            node *bf = other(other, closure_msg::tree_first{});
            node *bl = other(other, closure_msg::tree_last{});
            std::size_t an = self(self, closure_msg::tree_size{});
            std::size_t bn = other(other, closure_msg::tree_size{});
            self(self, closure_msg::tree_link_header{}, br, bf, bl, bn, bh);
            other(other, closure_msg::tree_link_header{}, ar, af, al, an,
                  ah);
        }
        // --- accessors -------------------------------------------------
        else if constexpr (std::same_as<T, closure_msg::tree_first>)
        {
            return static_cast<node *>(
                lens(&header)(closure_msg::next{}));
        }
        else if constexpr (std::same_as<T, closure_msg::tree_last>)
        {
            return static_cast<node *>(
                lens(&header)(closure_msg::prev{}));
        }
        else if constexpr (std::same_as<T, closure_msg::tree_root>)
        {
            return static_cast<node *>(
                lens(&header)(closure_msg::left{}));
        }
        else if constexpr (std::same_as<T, closure_msg::tree_end>)
        {
            return static_cast<node *>(&header);
        }
        else if constexpr (std::same_as<T, closure_msg::tree_header>)
        {
            return static_cast<node *>(&header);
        }
        else if constexpr (std::same_as<T, closure_msg::tree_next>)
        {
            node *p = std::get<0>(std::forward_as_tuple(args...));
            return static_cast<node *>(lens(p)(closure_msg::next{}));
        }
        else if constexpr (std::same_as<T, closure_msg::tree_prev>)
        {
            node *p = std::get<0>(std::forward_as_tuple(args...));
            return static_cast<node *>(lens(p)(closure_msg::prev{}));
        }
        else if constexpr (std::same_as<T, closure_msg::tree_size>)
        {
            return count;
        }
        else if constexpr (std::same_as<T, closure_msg::tree_comp_ref>)
        {
            return (comp);
        }
        else if constexpr (std::same_as<T, closure_msg::tree_swap_comp>)
        {
            auto &other = std::get<0>(std::forward_as_tuple(args...));
            using std::swap;
            swap(comp, other(other, closure_msg::tree_comp_ref{}));
        }
        else if constexpr (std::same_as<T, closure_msg::tree_set_comp>)
        {
            static_assert(sizeof...(args) == 1);
            comp = std::get<0>(
                std::forward_as_tuple(std::forward<decltype(args)>(args)...));
        }
        // --- iterator constructors ------------------------------------
        else if constexpr (std::same_as<T, closure_msg::tree_make_iter>)
        {
            node *p = std::get<0>(std::forward_as_tuple(args...));
            return make_iter(make_iter, make_citer, p);
        }
        else if constexpr (std::same_as<T,
                                        closure_msg::tree_make_citer>)
        {
            node *p = std::get<0>(std::forward_as_tuple(args...));
            return make_citer(make_citer, p);
        }
        else
        {
            static_assert(!sizeof(T), "unknown tree message");
        }
    };

    return tree;
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

    using node_t = std::remove_pointer_t<decltype(
        std::declval<core_t &>()(std::declval<core_t &>(),
                                 closure_msg::tree_root{}))>;
    using iter_fn = decltype(std::declval<core_t &>()(
        std::declval<core_t &>(), closure_msg::tree_make_iter{},
        std::declval<node_t *>()));
    using citer_fn = decltype(std::declval<core_t &>()(
        std::declval<core_t &>(), closure_msg::tree_make_citer{},
        std::declval<node_t *>()));
    using node_allocator_type =
        typename std::allocator_traits<Allocator>::template rebind_alloc<
            node_t>;
    using node_alloc_traits = std::allocator_traits<node_allocator_type>;
    using value_allocator_type =
        typename std::allocator_traits<Allocator>::template rebind_alloc<
            Value>;
    using value_alloc_traits = std::allocator_traits<value_allocator_type>;

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

        using fn_type = std::conditional_t<Const, citer_fn, iter_fn>;

        // The iterator's entire position is the closure.  There is no cursor
        // field: dereferencing, succeeding and preceding are messages sent to
        // the function.  This class is only the standard adapter that supplies
        // default construction and copy assignment required by the library;
        // `optional` is storage for the closure, not tree state.
        std::optional<fn_type> fn_{};

        constexpr explicit iterator_impl(fn_type fn) noexcept
            : fn_(std::move(fn)) {}

        constexpr void assign_from(const std::optional<fn_type> &other)
        {
            fn_.reset();
            if (other)
            {
                fn_.emplace(*other);
            }
        }

        constexpr node_t *node_ptr() const noexcept
        {
            return fn_ ? (*fn_)(closure_msg::iter_node{}) : nullptr;
        }

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
        constexpr iterator_impl(const iterator_impl &) = default;
        constexpr iterator_impl(iterator_impl &&) noexcept = default;

        constexpr iterator_impl &operator=(const iterator_impl &other)
        {
            if (this != &other)
            {
                assign_from(other.fn_);
            }
            return *this;
        }

        constexpr iterator_impl &operator=(iterator_impl &&other)
        {
            if (this != &other)
            {
                fn_.reset();
                if (other.fn_)
                {
                    fn_.emplace(std::move(*other.fn_));
                }
            }
            return *this;
        }

        template <bool C>
            requires(Const || !C)
        constexpr iterator_impl(const iterator_impl<C> &other)
        {
            if (!other.fn_)
            {
                return;
            }
            if constexpr (C == Const)
            {
                fn_.emplace(*other.fn_);
            }
            else
            {
                // Turning an iterator into a const_iterator is itself an
                // iterator message; the answer is another iterator function.
                fn_.emplace((*other.fn_)(closure_msg::iter_const{}));
            }
        }

        constexpr reference operator*() const noexcept
        {
            return (*fn_)(closure_msg::iter_value{});
        }
        constexpr pointer operator->() const noexcept
        {
            return std::addressof(operator*());
        }

        constexpr iterator_impl &operator++() noexcept
        {
            fn_.emplace((*fn_)(closure_msg::iter_next{}));
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
            fn_.emplace((*fn_)(closure_msg::iter_prev{}));
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
            if (!a.fn_ || !b.fn_)
            {
                return !a.fn_ && !b.fn_;
            }
            return (*a.fn_)(closure_msg::iter_equal{}, *b.fn_);
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
            if (p->closure_engaged())
            {
                value_allocator_type va(a);
                value_alloc_traits::destroy(va, p->closure_value());
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
            return *p_->closure_value();
        }

        constexpr key_type &key() const noexcept
            requires Mapped
        {
            return const_cast<key_type &>(KeyOfValue{}(*p_->closure_value()));
        }

        constexpr mapped_type &mapped() const noexcept
            requires Mapped
        {
            return p_->closure_value()->second;
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
        send(closure_msg::tree_reset{});
    }

    // The wrapper is only an adapter: every operation is a message to the
    // tree function.  The first argument makes the closure apply itself.
    template <class Tag, class... Args>
    constexpr decltype(auto) send(Tag tag, Args &&...args) const
    {
        return core_(core_, tag, std::forward<Args>(args)...);
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
        if (p->closure_engaged())
        {
            value_allocator_type va(a);
            value_alloc_traits::destroy(va, p->closure_value());
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
                va, p->closure_value(), std::forward<Args>(args)...);
            p->closure_set_engaged(true);
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
        if (send(closure_msg::tree_size{}) != 0)
        {
            node_t *r = send(closure_msg::tree_root{});
            send(closure_msg::tree_clear{}, r,
                 [this](node_t *p)
                 { destroy_node(p); });
        }
        send(closure_msg::tree_reset{});
    }

    template <bool Move>
    void clone_from(basic_ordered_tree &other)
    {
        auto create = [this](node_t *src) -> node_t *
        {
            if constexpr (Move)
            {
                return create_node(std::move(*src->closure_value()));
            }
            else
            {
                return create_node(*src->closure_value());
            }
        };
        auto destroy = [this](node_t *p)
        { destroy_node(p); };
        node_t *r = send(closure_msg::tree_clone{},
                         other.send(closure_msg::tree_root{}), create,
                         destroy);
        send(closure_msg::tree_adopt{}, r,
             other.send(closure_msg::tree_size{}));
    }

    template <bool Const>
    constexpr iterator_impl<Const> iter_at(node_t *p) const noexcept
    {
        if constexpr (Const)
        {
            return iterator_impl<Const>(
                send(closure_msg::tree_make_citer{}, p));
        }
        else
        {
            return iterator_impl<Const>(
                send(closure_msg::tree_make_iter{}, p));
        }
    }

    template <bool Const>
    constexpr iterator_impl<Const> begin_impl() const noexcept
    {
        return iter_at<Const>(send(closure_msg::tree_first{}));
    }

    template <bool Const>
    constexpr iterator_impl<Const> end_impl() const noexcept
    {
        return iter_at<Const>(send(closure_msg::tree_end{}));
    }

    template <class... Args>
    value_insert_return emplace_impl(Args &&...args)
    {
        node_t *p = create_node(std::forward<Args>(args)...);
        node_t *pos = nullptr;
        bool inserted = false;
        try
        {
            auto result = send(closure_msg::tree_insert_node{}, p);
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
            auto result = send(closure_msg::tree_insert_node_hint{}, p,
                               hint.node_ptr());
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
        return iter_at<Const>(send(closure_msg::tree_lower_bound{}, k));
    }

    template <bool Const, class K>
    iterator_impl<Const> upper_bound_impl(const K &k) const
    {
        return iter_at<Const>(send(closure_msg::tree_upper_bound{}, k));
    }

    template <bool Const, class K>
    iterator_impl<Const> find_impl(const K &k) const
    {
        return iter_at<Const>(send(closure_msg::tree_find_node{}, k));
    }

    template <bool Const, class K>
    std::pair<iterator_impl<Const>, iterator_impl<Const>>
    equal_range_impl(const K &k) const
    {
        return {iter_at<Const>(send(closure_msg::tree_lower_bound{}, k)),
                iter_at<Const>(send(closure_msg::tree_upper_bound{}, k))};
    }

    template <class K>
    size_type count_impl(const K &k) const
    {
        node_t *first = send(closure_msg::tree_lower_bound{}, k);
        node_t *last = send(closure_msg::tree_upper_bound{}, k);
        size_type n = 0;
        while (first != last)
        {
            ++n;
            first = send(closure_msg::tree_next{}, first);
        }
        return n;
    }

    template <class K>
    size_type erase_key_impl(const K &k)
    {
        node_t *first = send(closure_msg::tree_lower_bound{}, k);
        node_t *last = send(closure_msg::tree_upper_bound{}, k);
        size_type n = 0;
        while (first != last)
        {
            node_t *next = send(closure_msg::tree_next{}, first);
            send(closure_msg::tree_detach{}, first);
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
              init_tag{}, other.send(closure_msg::tree_comp_ref{}),
              std::allocator_traits<
                  Allocator>::select_on_container_copy_construction(other.alloc_))
    {
        clone_from<false>(const_cast<basic_ordered_tree &>(other));
    }

    basic_ordered_tree(basic_ordered_tree &&other) noexcept(
        std::allocator_traits<Allocator>::is_always_equal::value &&
        std::is_nothrow_move_constructible_v<Compare>)
        : basic_ordered_tree(
              init_tag{},
              std::move(other.send(closure_msg::tree_comp_ref{})),
              std::move(other.alloc_))
    {
        send(closure_msg::tree_take_from{}, other.core_);
    }

    basic_ordered_tree(const basic_ordered_tree &other,
                       const std::type_identity_t<Allocator> &alloc)
        : basic_ordered_tree(init_tag{},
                             other.send(closure_msg::tree_comp_ref{}), alloc)
    {
        clone_from<false>(const_cast<basic_ordered_tree &>(other));
    }

    basic_ordered_tree(basic_ordered_tree &&other,
                       const std::type_identity_t<Allocator> &alloc)
        : basic_ordered_tree(
              init_tag{},
              std::move(other.send(closure_msg::tree_comp_ref{})), alloc)
    {
        if (alloc_ == other.alloc_)
        {
            send(closure_msg::tree_take_from{}, other.core_);
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
        send(closure_msg::tree_comp_ref{}) =
            other.send(closure_msg::tree_comp_ref{});
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
        send(closure_msg::tree_comp_ref{}) =
            std::move(other.send(closure_msg::tree_comp_ref{}));
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
            send(closure_msg::tree_take_from{}, other.core_);
        }
        else
        {
            if (alloc_ == other.alloc_)
            {
                clear_nodes();
                send(closure_msg::tree_take_from{}, other.core_);
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
    [[nodiscard]] bool empty() const noexcept
    {
        return send(closure_msg::tree_size{}) == 0;
    }
    [[nodiscard]] size_type size() const noexcept
    {
        return send(closure_msg::tree_size{});
    }
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
            const key_type &k = KeyOfValue{}(nh.value());
            auto it = find(k);
            if (it != end())
            {
                return {it, false, std::move(nh)};
            }
        }

        if (alloc_ == nh.alloc_)
        {
            node_t *p = nh.p_;
            auto [pos, inserted] = send(closure_msg::tree_insert_node{}, p);
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
            node_t *p = create_node(std::move(nh.value()));
            node_t *pos = nullptr;
            bool inserted = false;
            try
            {
                auto result = send(closure_msg::tree_insert_node{}, p);
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
            auto it = find(KeyOfValue{}(nh.value()));
            if (it != end())
            {
                return {it, false, node_type{}};
            }
        }
        node_t *p = create_node(nh.value());
        node_t *pos = nullptr;
        bool inserted = false;
        try
        {
            auto result = send(closure_msg::tree_insert_node{}, p);
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
            auto result = send(closure_msg::tree_insert_node_hint{}, p, hint.node_ptr());
            if (result.second)
            {
                nh.p_ = nullptr;
            }
            return iter_at<Mapped ? false : true>(result.first);
        }
        if constexpr (!AllowDuplicates)
        {
            auto it = find(KeyOfValue{}(nh.value()));
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
        node_t *p = pos.node_ptr();
        node_t *n = send(closure_msg::tree_next{}, p);
        send(closure_msg::tree_detach{}, p);
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
        node_t *stop = last.node_ptr();
        node_t *cur = first.node_ptr();
        while (cur != stop)
        {
            node_t *n = send(closure_msg::tree_next{}, cur);
            send(closure_msg::tree_detach{}, cur);
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
        send(closure_msg::tree_swap_comp{}, other.core_);
        send(closure_msg::tree_swap_trees{}, other.core_);
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
        node_t *p = pos.node_ptr();
        node_type result(alloc_);
        send(closure_msg::tree_detach{}, p);
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
        node_t *p = send(closure_msg::tree_find_node{}, key);
        if (p == send(closure_msg::tree_end{}))
        {
            return node_type{};
        }
        node_type result(alloc_);
        send(closure_msg::tree_detach{}, p);
        result.p_ = p;
        return result;
    }

    template <class K>
        requires(is_transparent<Compare>::value &&
                 !std::convertible_to<K, iterator> &&
                 !std::convertible_to<K, const_iterator>)
    node_type extract(K &&key)
    {
        node_t *p = send(closure_msg::tree_find_node{}, key);
        if (p == send(closure_msg::tree_end{}))
        {
            return node_type{};
        }
        node_type result(alloc_);
        send(closure_msg::tree_detach{}, p);
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
    key_compare key_comp() const
    {
        return send(closure_msg::tree_comp_ref{});
    }

    value_compare value_comp() const
    {
        if constexpr (Mapped)
        {
            return value_compare(send(closure_msg::tree_comp_ref{}));
        }
        else
        {
            return send(closure_msg::tree_comp_ref{});
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
            auto result = send(closure_msg::tree_insert_node{}, p);
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
            auto result = send(closure_msg::tree_insert_node{}, p);
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
            auto result = send(closure_msg::tree_insert_node{}, p);
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
        auto [pos, inserted] = send(closure_msg::tree_insert_node{}, p);
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

    // The local engine type is a Voldemort function type, yet it can be
    // persisted as a field of a local class thanks to decltype.
    auto local_factory = make_stl_toolkit;
    using local_core_t = decltype(local_factory.template operator()<int, int, std::identity,
                                                                    std::less<int>, false>(
        std::less<int>{}));
    static_assert(!std::is_copy_constructible_v<local_core_t>);
    static_assert(std::is_move_constructible_v<local_core_t>);
    struct holder
    {
        local_core_t engine;
    };
    holder h{local_factory.template operator()<int, int, std::identity,
                                               std::less<int>, false>(
        std::less<int>{})};
    h.engine(h.engine, closure_msg::tree_reset{});
    assert(h.engine(h.engine, closure_msg::tree_size{}) == 0);
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
