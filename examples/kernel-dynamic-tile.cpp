#include "RAJA/RAJA.hpp"

int main(int RAJA_UNUSED_ARG(argc), char** RAJA_UNUSED_ARG(argv[]))
{
  std::cout << "\n\nRAJA dynamic_tile example...\n\n";

  // Using policy = KernelPolicy<Tile<tile_dynamic<0>, seq_exec, …>>;
  // RAJA::kernel_param<policy>(
  //  make_tuple(RangeSegment(0,N)),
  //   make_tuple(32),  // param 0 is referenced by tile_dynamic
  //   [=](int i, int tile_size){
  //
  //   });

  using namespace RAJA;

  kernel_param<KernelPolicy<statement::Tile<
      1, tile_dynamic<1>, seq_exec,
      statement::Tile<
          0, tile_dynamic<0>, seq_exec,
          statement::For<1, seq_exec,
                         statement::For<0, seq_exec, statement::Lambda<0>>>>>>>(
      make_tuple(RangeSegment{0, 25}, RangeSegment{0, 25}),
      make_tuple(TileSize{5}, TileSize{10}),
      // make_tuple(TileSize(10)), // not sure we need this, good for
      // static_assert
      [=](int i, int j, TileSize x, TileSize y)
      {
        std::cout << "Running index (" << i << "," << j << ") of " << x.size
                  << "x" << y.size << " tile." << std::endl;
      });
}
