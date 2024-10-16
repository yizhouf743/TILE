// Authors: Kanav Gupta, Neha Jawalkar
// Copyright:
// 
// Copyright (c) 2024 Microsoft Research
// 
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#pragma once
#include <sytorch/backend/baseline_cleartext.h>

template <typename T>
class PiranhaClearText : public BaselineClearText<T> {
public:
    T pirhana_inverse(T x, u64 scale)
    {
        T alpha = 0;
        while((1LL << alpha) < x) {
            alpha++;
        }
        // inv(x/2^scale) * 2^scale = [inv(x/2^alpha) * 2^alpha] * 2^(scale - alpha) * 2^(scale - alpha)
        T a_alpha = 2.63 * (1LL << alpha);
        T b_alpha = -5.857 * (1LL << alpha);
        T c_alpha = 4.245 * (1LL << alpha);

        T res = a_alpha * x + b_alpha * (1LL << alpha);
        res >>= alpha;
        res = res * x + c_alpha * (1LL << alpha);
        res >>= alpha;
        if (scale > alpha) {
            res = res * (1LL<<(scale-alpha));
            res = res * (1LL<<(scale-alpha));
        }
        else {
            res = res / (1LL<<(alpha-scale));
            res = res / (1LL<<(alpha-scale));
        }
        return res;
    }
    
    
    void softmax(Tensor<T> &_in, Tensor<T> &_out, u64 scale, u64 mode)
    {
        always_assert(_in.shape.size() == 2);
        always_assert(_out.shape.size() == 2);
        always_assert(_in.shape[0] == _out.shape[0]);
        always_assert(_in.shape[1] == _out.shape[1]);
        T twofix = 2 * (1LL << scale);

        auto in = _in.as_2d();
        auto out = _out.as_2d();

        auto batchSize = in.d1;
        auto numClasses = in.d2;
        T exps[numClasses];
        for(int b = 0; b < batchSize; ++b) {
            T max = in(b, 0);
            for(u64 j = 1; j < numClasses; ++j) {
                if(in(b, j) > max) {
                    max = in(b, j);
                }
            }

            T den = 0;
            for(u64 j = 0; j < numClasses; ++j) {
                T x = max - in(b, j);
                exps[j] = (x < twofix ? twofix - x : 0) / 2;
                den += exps[j];
            }

            T inv_den = pirhana_inverse(den, scale);
            for(u64 j = 0; j < numClasses; ++j) {
                out(b, j) = exps[j] * inv_den;
                out(b, j) >>= scale;
            }
        }
    }

    void softmax_triangular(Tensor<T> &_in, Tensor<T> &_out, u64 scale, u64 mode)
    {
        always_assert(_in.shape.size() == 2);
        always_assert(_out.shape.size() == 2);
        always_assert(_in.shape[0] == _out.shape[0]);
        always_assert(_in.shape[1] == _out.shape[1]);
        always_assert(_in.shape[0] == _in.shape[1]); // should be a square matrix
        T twofix = 2 * (1LL << scale);

        auto in = _in.as_2d();
        auto out = _out.as_2d();

        auto batchSize = in.d1;
        auto numClasses = in.d2;
        T exps[numClasses];

        out.zero();
        out(0, 0) = T(1LL << (scale));
        
        for(int b = 1; b < batchSize; ++b) {
            T max = in(b, 0);
            for(u64 j = 1; j < b + 1; ++j) {
                if(in(b, j) > max) {
                    max = in(b, j);
                }
            }

            T den = 0;
            for(u64 j = 0; j < b + 1; ++j) {
                T x = max - in(b, j);
                exps[j] = (x < twofix ? twofix - x : 0) / 2;
                // exps[j] = T(std::exp(-x / double(1LL<<scale)) * (1LL<<scale));
                den += exps[j];
            }

            T inv_den = pirhana_inverse(den, scale);
            for(u64 j = 0; j < b + 1; ++j) {
                out(b, j) = exps[j] * inv_den;
                out(b, j) >>= scale;
            }
        }
    }

};
