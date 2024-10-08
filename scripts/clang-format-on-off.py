import string
import re
import os
from argparse import ArgumentParser

# Parse a list of strings (that is assumed to be valid cpp code).  Identify all nested 
# templated type definitions of depth greater than cutoff.  Return a list of integers
# representing the indexes of such declarations
def find_nested_template_declarations(inp : list[string], cutoff : int = 2) -> list[tuple[int,int]]:
    out = []
    i = 0
    while i < len(inp):
        line = inp[i]
        ptr1 = i
        ptr2 = i
        num_brackets = len(re.findall(r'(?<!<)<(?!<| )', inp[ptr1]))
        num_closing_brackets = len(re.findall(r'(?<!>)>(?!>| )', inp[ptr1]))
        num_braces = 0
        num_closing_braces = 0
        num_paren_open = inp[ptr1].count('(')
        num_paren_close = inp[ptr1].count(')')
        # bool for multiline template class invocation
        is_template_decl = (
                            (num_brackets > num_closing_brackets or
                            (num_brackets > 0 and ';' not in inp[ptr1]) or
                            num_paren_open > num_paren_close) and
                            ";" not in inp[ptr1] and
                            "include" not in inp[ptr1])
        while is_template_decl and ptr2 < len(inp):
            #print(i, num_brackets, num_closing_brackets)
            num_brackets += len(re.findall(r'(?<!<)<(?!<)', inp[ptr2]))
            # num_closing_brackets += inp[ptr2].count(">")
            num_braces += inp[ptr2].count("{")
            num_closing_braces += inp[ptr2].count("}")
            # scroll till we get through all the the brackets and the declaration
            if ";" in inp[ptr2]:
                #print("kword", ptr2, num_brackets)
                if num_braces == num_closing_braces:
                 #   print("kword", inp[ptr2], ptr2, num_brackets)
                    break
            ptr2 += 1
        # count the depth of template parametrization
        if num_brackets > cutoff and ptr1 != ptr2:
            out.append((ptr1, ptr2))
        i = ptr2 + 1

    return out

def add_clang_format_comments(inp : list[string], insertions: list[tuple[int,int]]) -> list[string]:
    clang_format_off = "// clang-format off\n"
    clang_format_on = "// clang-format on\n"
    num_insertions = 0
    current_insertion_idx = 0
    out = []
    for i in range(0, len(inp)):
        done_inserting = current_insertion_idx >= len(insertions)
        if not done_inserting and i == insertions[current_insertion_idx][0]:
            out.append(clang_format_off)
        out.append(inp[i])
        if not done_inserting and i == insertions[current_insertion_idx][1] + 1:
            out.append(clang_format_on)
            # we've handled one off-on pair, increment pointer in insertion array
            current_insertion_idx += 1
        
    return out

def refactor_file(fname, str):
    with open(fname, "w") as f:
        f.write(str)
         
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-d", "--directory", dest="directory", help="directory to refactor")
    args = parser.parse_args()
    dir = args.directory
    if args.directory is None or not os.path.exists(dir):
        exit("please specify a valid directory to add clang format annotations")
    for d, _, files in os.walk(dir):
        for f in files:
            out = ""
            with open(os.path.abspath(os.path.join(d, f)), "r") as f_obj:
                f_list = f_obj.readlines()
                out = add_clang_format_comments(f_list, find_nested_template_declarations(f_list))
            with open(os.path.abspath(os.path.join(d, f)), "w") as f_obj:
                for line in out:
                    f_obj.write(f"{line}")

#
    #test_str = """
    #/*
    #  Define CUDA/HIP matrix multiplication kernel for comparison to RAJA version
    #*/
    ##if defined(RAJA_ENABLE_CUDA) || defined(RAJA_ENABLE_HIP)
    #__global__ void matMultKernel(int N, double* C, double* A, double* B)
    #{
    #  int row = blockIdx.y * blockDim.y + threadIdx.y;
    #  int col = blockIdx.x * blockDim.x + threadIdx.x;
#
    #  if ( row < N && col < N ) {
    #    double dot = 0.0;
    #    for (int k = 0; k < N; ++k) {
    #      dot += A(row, k) * B(k, col);
    #    }
#
    #    C(row, col) = dot;
    #  }
    #}
#
    #__global__ void sharedMatMultKernel(int N, double* C, double* A, double* B)
    #{
#
    #  int Row = blockIdx.y*THREAD_SZ + threadIdx.y;
    #  int Col = blockIdx.x*THREAD_SZ + threadIdx.x;
#
    #  __shared__ double As[THREAD_SZ][THREAD_SZ];
    #  __shared__ double Bs[THREAD_SZ][THREAD_SZ];
    #  __shared__ double Cs[THREAD_SZ][THREAD_SZ];
#
    #  Cs[threadIdx.y][threadIdx.x] = 0.0;
#
    #  for (int k = 0; k < (THREAD_SZ + N - 1)/THREAD_SZ; k++) {
#
    #    if ( static_cast<int>(k*THREAD_SZ + threadIdx.x) < N && Row < N )
    #      As[threadIdx.y][threadIdx.x] = A[Row*N + k*THREAD_SZ + threadIdx.x];
    #    else
    #      As[threadIdx.y][threadIdx.x] = 0.0;
#
    #    if ( static_cast<int>(k*THREAD_SZ + threadIdx.y) < N && Col < N)
    #      Bs[threadIdx.y][threadIdx.x] = B[(k*THREAD_SZ + threadIdx.y)*N + Col];
    #    else
    #      Bs[threadIdx.y][threadIdx.x] = 0.0;
#
    #    __syncthreads();
#
    #    for (int n = 0; n < THREAD_SZ; ++n)
    #      Cs[threadIdx.y][threadIdx.x] += As[threadIdx.y][n] * Bs[n][threadIdx.x];
#
    #    __syncthreads();
    #  }
#
    #  if (Row < N && Col < N)
    #    C[((blockIdx.y * blockDim.y + threadIdx.y)*N) +
    #      (blockIdx.x * blockDim.x)+ threadIdx.x] = Cs[threadIdx.y][threadIdx.x];
    #}
    ##endif
    #"""
    #f_list = test_str.split('\n')
    #"\n".join(add_clang_format_comments(f_list, find_nested_template_declarations(f_list)))