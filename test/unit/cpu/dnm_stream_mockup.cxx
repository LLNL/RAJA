#include <RAJA/RAJA.hxx>

#include <RAJA/Stream.hxx>

int main(){
  RAJA::stream_pool dog_pool(8);
  dog_pool.run(0, [=](){
    std::cout<<"DOGS ARE PRETTY COOL I FOR ONE BELIEVE\n";
  });
  dog_pool.run(1, [=](){
    std::cout<<"DOGS ARE PRETTY COOL I FOR ONE BELIEVE\n";
  });
  dog_pool.run(0,[=](){
    std::cout<<"DOGS ARE DEF PRETTY COOL I FOR ONE BELIEVE\n";
  });
}


