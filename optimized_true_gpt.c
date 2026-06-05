#include <stdint.h>

int64_t predict_next_token_optimized(const int64_t* context, int64_t context_len, int64_t query) {
    int64_t y = 0;
    for (int64_t j = 0; j < context_len - 1; ++j) {
        int64_t diff = ~(~(query ^ context[j]));
        int64_t any_diff = (diff | (diff >> 1) | (diff >> 2) | (diff >> 3)) & 1;
        int64_t mask = -(any_diff ^ 1);
        y |= mask & context[j+1];
    }
    return y;
}
