// Exact-order CPU inference for the deployable Python policies.
//
// The Python reference computes every affine layer as
// ``sum(map(mul, row, x)) + bias`` (left-to-right double additions) and then
// applies libm ``tanh``/``exp``.  This translation keeps the same operation
// order and is compiled with FMA contraction, reassociation and vectorisation
// disabled, so results are bit-identical to the Python path.
#include <cmath>
#include <cstddef>
#include <vector>

// CPython 3.12+ ``sum()`` uses Neumaier compensated summation for floats.
// Reproduce that algorithm exactly; a naive left-to-right accumulator is not
// bit-identical to the reference.
static inline double dot(const double* w, const double* x, int n) {
    double total = 0.0;
    double compensation = 0.0;
    for (int i = 0; i < n; ++i) {
        const double value = w[i] * x[i];
        const double next = total + value;
        if (std::fabs(total) >= std::fabs(value)) {
            compensation += (total - next) + value;
        } else {
            compensation += (value - next) + total;
        }
        total = next;
    }
    return total + compensation;
}

static inline double sigmoid(double x) {
    if (x >= 0.0) {
        return 1.0 / (1.0 + std::exp(-std::fmin(x, 40.0)));
    }
    const double z = std::exp(std::fmax(x, -40.0));
    return z / (1.0 + z);
}

// One MLP forward pass.  ``dims`` stores (out_size, in_size) per layer and
// ``w`` stores row-major weights followed by that layer's biases.
static double mlp_forward(const double* x, const double* w,
                          const int* dims, int n_layers,
                          std::vector<double>& buf0,
                          std::vector<double>& buf1) {
    const double* cur = x;
    double* out = buf0.data();
    double* other = buf1.data();
    const double* cursor = w;
    for (int layer = 0; layer < n_layers; ++layer) {
        const int out_n = dims[2 * layer];
        const int in_n = dims[2 * layer + 1];
        const double* bias = cursor + static_cast<std::size_t>(out_n) * in_n;
        for (int o = 0; o < out_n; ++o) {
            double value = dot(cursor + static_cast<std::size_t>(o) * in_n,
                               cur, in_n) + bias[o];
            if (layer + 1 < n_layers) {
                value = std::tanh(value);
            }
            out[o] = value;
        }
        cursor = bias + out_n;
        if (layer + 1 < n_layers) {
            cur = out;
            double* next = other;
            other = out;
            out = next;
        }
    }
    return out[0];
}

static int max_width(const int* dims, int n_layers) {
    int width = 1;
    for (int i = 0; i < n_layers; ++i) {
        if (dims[2 * i] > width) {
            width = dims[2 * i];
        }
        if (dims[2 * i + 1] > width) {
            width = dims[2 * i + 1];
        }
    }
    return width;
}

// Advance one or more GRU layers in place and return the final GRU output in
// ``cur``.  ``h`` is a [n_layers, hidden] row-major buffer.
static void gru_core(const double* x, int n_in, double* h, int n_layers,
                     int hidden, const double* w,
                     std::vector<double>& cur,
                     std::vector<double>& gi,
                     std::vector<double>& gh,
                     std::vector<double>& new_h) {
    cur.assign(x, x + n_in);
    const double* cursor = w;
    const int gate = 3 * hidden;
    int layer_in = n_in;
    for (int layer = 0; layer < n_layers; ++layer) {
        const double* w_ih = cursor;
        const double* b_ih = w_ih + static_cast<std::size_t>(gate) * layer_in;
        const double* w_hh = b_ih + gate;
        const double* b_hh = w_hh + static_cast<std::size_t>(gate) * hidden;
        cursor = b_hh + gate;

        double* h_layer = h + static_cast<std::size_t>(layer) * hidden;
        for (int j = 0; j < gate; ++j) {
            gi[j] = dot(w_ih + static_cast<std::size_t>(j) * layer_in,
                        cur.data(), layer_in) + b_ih[j];
            gh[j] = dot(w_hh + static_cast<std::size_t>(j) * hidden,
                        h_layer, hidden) + b_hh[j];
        }
        for (int j = 0; j < hidden; ++j) {
            const double r = sigmoid(gi[j] + gh[j]);
            const double z = sigmoid(gi[hidden + j] + gh[hidden + j]);
            const double candidate = std::tanh(
                gi[2 * hidden + j] + r * gh[2 * hidden + j]);
            new_h[j] = (1.0 - z) * candidate + z * h_layer[j];
        }
        for (int j = 0; j < hidden; ++j) {
            h_layer[j] = new_h[j];
        }
        cur.assign(new_h.begin(), new_h.begin() + hidden);
        layer_in = hidden;
    }
}

extern "C" int pdi_mlp_forward(const double* x, const double* w,
                               const int* dims, int n_layers,
                               double* out) {
    if (n_layers <= 0 || x == nullptr || w == nullptr || dims == nullptr) {
        return -1;
    }
    const int width = max_width(dims, n_layers);
    std::vector<double> buf0(static_cast<std::size_t>(width));
    std::vector<double> buf1(static_cast<std::size_t>(width));
    *out = mlp_forward(x, w, dims, n_layers, buf0, buf1);
    return 0;
}

extern "C" int pdi_gru_step(const double* x, int n_in, double* h,
                            int n_layers, int hidden, const double* w,
                            const double* head_w, const int* head_dims,
                            int n_head_layers, double* out) {
    if (n_layers <= 0 || hidden <= 0 || x == nullptr || h == nullptr ||
        w == nullptr || head_w == nullptr || head_dims == nullptr) {
        return -1;
    }
    std::vector<double> cur;
    std::vector<double> gi(static_cast<std::size_t>(3 * hidden));
    std::vector<double> gh(static_cast<std::size_t>(3 * hidden));
    std::vector<double> new_h(static_cast<std::size_t>(hidden));
    gru_core(x, n_in, h, n_layers, hidden, w, cur, gi, gh, new_h);

    const int width = max_width(head_dims, n_head_layers);
    const int head_input = static_cast<int>(cur.size());
    const int buf_width = width > head_input ? width : head_input;
    std::vector<double> buf0(static_cast<std::size_t>(buf_width));
    std::vector<double> buf1(static_cast<std::size_t>(buf_width));
    *out = mlp_forward(cur.data(), head_w, head_dims, n_head_layers,
                       buf0, buf1);
    return 0;
}

extern "C" int pdi_selfplay_gru_step(
        const double* x, int n_in, double* h, int n_layers, int hidden,
        const double* w,
        const double* actor_w, const int* actor_dims, int n_actor_layers,
        const double* critic_w, const int* critic_dims, int n_critic_layers,
        double* out) {
    if (n_layers <= 0 || hidden <= 0 || x == nullptr || h == nullptr ||
        w == nullptr || actor_w == nullptr || actor_dims == nullptr ||
        critic_w == nullptr || critic_dims == nullptr || out == nullptr) {
        return -1;
    }
    std::vector<double> cur;
    std::vector<double> gi(static_cast<std::size_t>(3 * hidden));
    std::vector<double> gh(static_cast<std::size_t>(3 * hidden));
    std::vector<double> new_h(static_cast<std::size_t>(hidden));
    gru_core(x, n_in, h, n_layers, hidden, w, cur, gi, gh, new_h);

    int width = 1;
    for (int i = 0; i < n_actor_layers; ++i) {
        if (actor_dims[2 * i] > width) width = actor_dims[2 * i];
        if (actor_dims[2 * i + 1] > width) width = actor_dims[2 * i + 1];
    }
    for (int i = 0; i < n_critic_layers; ++i) {
        if (critic_dims[2 * i] > width) width = critic_dims[2 * i];
        if (critic_dims[2 * i + 1] > width) width = critic_dims[2 * i + 1];
    }
    const int head_input = static_cast<int>(cur.size());
    if (head_input > width) width = head_input;
    std::vector<double> buf0(static_cast<std::size_t>(width));
    std::vector<double> buf1(static_cast<std::size_t>(width));

    out[0] = mlp_forward(cur.data(), actor_w, actor_dims, n_actor_layers,
                         buf0, buf1);
    out[1] = mlp_forward(cur.data(), critic_w, critic_dims, n_critic_layers,
                         buf0, buf1);
    return 0;
}
