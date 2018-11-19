#include <stdio.h>
#include <torch/torch.h>

int main(){
    printf("Hello world");
    at::Tensor a = 3;
    IntList newSize = {1,2,3};
    a.expand(newSize);
    printf(a)
    return 0;
}