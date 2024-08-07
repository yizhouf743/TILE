#include "LinearHE/TILE-field.h"
#include "NonLinear/relu-field.h"
#include "library_fixed.h"
#include "plaintxt_operation.h"
#include <fstream>
#include <iostream>
#include <thread>

using namespace std;
using namespace seal;
using namespace sci;
using namespace seal::util;
using namespace std::chrono;

int party = 0;
int bitlength = 32;
int num_threads = 4;
int port = 8000;
string address = "127.0.0.1";
int image_h = 56;
int inp_chans = 64;
int filter_h = 3;
int out_chans = 64;
int pad_l = 0;
int pad_r = 0;
int stride = 2;
int filter_precision = 12;
int tile_type = 0;
float apply_ratio = 0.0;
int b = 4;
int batch_size = 0;


void field_relu_thread(int tid, uint64_t *z, uint64_t *x, int lnum_relu) {
  ReLUFieldProtocol<uint64_t> *relu_oracle;
  if (tid & 1) {
    relu_oracle = new ReLUFieldProtocol<uint64_t>(3 - party, FIELD,
                                                  iopackArr[tid], bitlength, b,
                                                  prime_mod, otpackArr[tid]);
  } else {
    relu_oracle = new ReLUFieldProtocol<uint64_t>(
        party, FIELD, iopackArr[tid], bitlength, b, prime_mod, otpackArr[tid]);
  }
  if (batch_size) {
    for (int j = 0; j < lnum_relu; j += batch_size) {
      if (batch_size <= lnum_relu - j) {
        relu_oracle->relu(z + j, x + j, batch_size);
      } else {
        relu_oracle->relu(z + j, x + j, lnum_relu - j);
      }
    }
  } else {
    relu_oracle->relu(z, x, lnum_relu);
  }

  delete relu_oracle;
  return;
}


void Conv(TILEField &he_conv, int32_t H, int32_t CI, int32_t FH, int32_t CO, int32_t tp, float ar, int32_t zPadHLeft, int32_t zPadHRight, int32_t strideH) {
  int newH = 1 + (H + zPadHLeft + zPadHRight - FH) / strideH;
  int N = 1;
  int W = H;
  int FW = FH;
  int zPadWLeft = zPadHLeft;
  int zPadWRight = zPadHRight;
  int strideW = strideH;
  int newW = newH;

  vector<vector<vector<vector<uint64_t>>>> filterArr(FH);
  vector<vector<vector<vector<uint64_t>>>> outArr(N);

  PRG128 prg;
  for (int i = 0; i < N; i++) {
    outArr[i].resize(newH);
    for (int j = 0; j < newH; j++) {
      outArr[i][j].resize(newW);
      for (int k = 0; k < newW; k++) {
        outArr[i][j][k].resize(CO);
      }
    }
  }
  if (party == ALICE) {
    for (int i = 0; i < FH; i++) {
      filterArr[i].resize(FW);
      for (int j = 0; j < FW; j++) {
        filterArr[i][j].resize(CI);
        for (int k = 0; k < CI; k++) {
          filterArr[i][j][k].resize(CO);
          prg.random_data(filterArr[i][j][k].data(), CO * sizeof(uint64_t));
          for (int h = 0; h < CO; h++) {
            filterArr[i][j][k][h] = ((int64_t)filterArr[i][j][k][h]) >> (64 - filter_precision);
          }
        }
      }
    }
  }

  vector<vector<vector<vector<uint64_t>>>> inputArr(N);
  for (int i = 0; i < N; i++) {
    inputArr[i].resize(H);
    for (int j = 0; j < H; j++) {
      inputArr[i][j].resize(W);
      for (int k = 0; k < W; k++) {
        inputArr[i][j][k].resize(CI);
        prg.random_mod_p<uint64_t>(inputArr[i][j][k].data(), CI, prime_mod);
      }
    }
  }

  // In this demo, We apply all feature and channel in tile strucuture to verify 
  // the correctness on Private infernce computation on Cryptflow2. This demo don't consider the
  // correctness on the case that input cannot full apply tile strucutre.

  // Internal tile setup:
  if (tp == 0){
    for (int i = 0; i < N; i++) {
      for (int j = 0; j < H; j++) {
        for (int k = 0; k < W; k++) {
          for (int h = 0; h < CI; h++) {
          if (int (k % 2) == 0 && int(j % 2) == 0){
                inputArr[i][j+1][k][h] = inputArr[i][j][k][h];
                inputArr[i][j+1][k+1][h] = inputArr[i][j][k][h];
                inputArr[i][j][k+1][h] = inputArr[i][j][k][h];
              }}
          }
      }
    }
  }
  else{
    // External tile setup:
    for (int i = 0; i < N; i++) {
      for (int j = 0; j < H; j++) {
        for (int k = 0; k < W; k++) {
          for (int h = 0; h < CI; h++) {
          if (int (h % 2) == 0){
                inputArr[i][j][k][h+1] = inputArr[i][j][k][h];
              }}
          }
      }
    }
  }

  uint64_t comm_start = he_conv.io->counter;
  INIT_TIMER;
  START_TIMER;
  he_conv.tile_convolution(N, H, W, CI, FH, FW, CO, tp, ar, zPadHLeft, zPadHRight, zPadWLeft,
                      zPadWRight, strideH, strideW, inputArr, filterArr, outArr,
                      true, true);
  STOP_TIMER("Total Time for Conv");
  uint64_t comm_end = he_conv.io->counter;
  cout << "Total Comm: " << (comm_end - comm_start) / (1.0 * (1ULL << 20))
       << endl;
}

// This is a demo to verify the functional correctness of apply tile structure in relu-conv module, 
// which is the basic unit on vgg and resnet.  
int main(int argc, char **argv) {
  ArgMapping amap;
  amap.arg("r", party, "Role of party: ALICE = 1; BOB = 2");
  amap.arg("nt", num_threads, "Number of Threads");
  amap.arg("l", bitlength, "Bitlength");
  amap.arg("ip", address, "IP Address of server (ALICE)");
  amap.arg("p", port, "Port Number");
  amap.arg("h", image_h, "Image Height/Width");
  amap.arg("f", filter_h, "Filter Height/Width");
  amap.arg("i", inp_chans, "Input Channels");
  amap.arg("o", out_chans, "Ouput Channels");
  amap.arg("s", stride, "stride");
  amap.arg("pl", pad_l, "Left Padding");
  amap.arg("pr", pad_r, "Right Padding");
  amap.arg("fp", filter_precision, "Filter Precision");
  amap.arg("tp", tile_type, "tile type to apply on ciphertext: Internal Tile = 0; External Tile = 1");
  amap.arg("ar", apply_ratio, "percentage of feature map(0)/channel(1) can apply Tile");
  amap.parse(argc, argv);
  prime_mod = sci::default_prime_mod.at(bitlength);

  cout << "=================================================================="
       << endl;
  cout << "Role: " << party << " - Bitlength: " << bitlength
       << " - Mod: " << prime_mod << " - Image: " << image_h << "x" << image_h
       << "x" << inp_chans << " - Filter: " << filter_h << "x" << filter_h
       << "x" << out_chans << "\n- Stride: " << stride << "x" << stride
       << " - Padding: " << pad_l << "x" << pad_r
       << " - # Threads: " << num_threads << endl;
  cout << "=================================================================="
       << endl;

  NetIO *io = new NetIO(party == 1 ? nullptr : address.c_str(), port);
  TILEField TILEField(party, io);
  Conv(TILEField, image_h, inp_chans, filter_h, out_chans, tile_type, apply_ratio, pad_l, pad_r, stride);
  
  // Estimate Performance on ReLU:
  int num_relu;
  int base_relu = inp_chans * image_h * image_h;
  auto ratio = static_cast<float>(apply_ratio);

  if (tile_type == 0){
    num_relu = floor(base_relu * ratio / 4) + floor(base_relu * (1- ratio));
  }
  else{
    num_relu = floor(base_relu * ratio / 2) + floor(base_relu * (1- ratio));
  }
  cout << base_relu << " x " << num_relu << endl;

  sci::PRG128 prg;
  uint64_t mask_l;
  if (bitlength == 64)
    mask_l = -1;
  else
    mask_l = (1ULL << bitlength) - 1;
  uint64_t *x = new uint64_t[num_relu];
  uint64_t *z = new uint64_t[num_relu];
  prg.random_mod_p<uint64_t>(x, num_relu, prime_mod);

  /********** Setup IO and Base OTs ***********/
  /********************************************/

  for (int i = 0; i < num_threads; i++) {
    iopackArr[i] = new IOPack(party, port + i, address);
    if (i & 1) {
      otpackArr[i] = new OTPack(iopackArr[i], 3 - party);
    } else {
      otpackArr[i] = new OTPack(iopackArr[i], party);
    }
  }
  std::cout << "All Base OTs Done" << std::endl;

  /************** Fork Threads ****************/
  /********************************************/

  uint64_t comm_sent = 0;
  uint64_t multiThreadedIOStart[num_threads];
  for (int i = 0; i < num_threads; i++) {
    multiThreadedIOStart[i] = iopackArr[i]->get_comm();
  }
  auto start = clock_start();
  std::thread relu_threads[num_threads];
  int chunk_size = num_relu / num_threads;
  for (int i = 0; i < num_threads; ++i) {
    int offset = i * chunk_size;
    int lnum_relu;
    if (i == (num_threads - 1)) {
      lnum_relu = num_relu - offset;
    } else {
      lnum_relu = chunk_size;
    }
    relu_threads[i] =
        std::thread(field_relu_thread, i, z + offset, x + offset, lnum_relu);
  }
  for (int i = 0; i < num_threads; ++i) {
    relu_threads[i].join();
  }

  long long t = time_from(start);
  for (int i = 0; i < num_threads; i++) {
    auto curComm = (iopackArr[i]->get_comm()) - multiThreadedIOStart[i];
    comm_sent += curComm;
  }
  std::cout << "Comm. Sent/ell: "
            << double(comm_sent * 8) / (bitlength * num_relu) << std::endl;

  delete[] x;
  delete[] z;

  /**** Process & Write Benchmarking Data *****/
  /********************************************/

  cout << "Number of ReLU/s:\t" << (double(num_relu) / t) * 1e6 << std::endl;
  cout << "ReLU Time (bitlength=" << bitlength << "; b=" << b << ")\t" << t
       << " mus" << endl;

  /******************* Cleanup ****************/
  /********************************************/

  for (int i = 0; i < num_threads; i++) {
    delete iopackArr[i];
    delete otpackArr[i];
  }
  return 0;
}
