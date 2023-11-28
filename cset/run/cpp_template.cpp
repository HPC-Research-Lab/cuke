#include <torch/extension.h>

inline size_t SetIntersection(int& first_arr, size_t first_size, int& second_arr, size_t second_size, int& res_arr)
{
  int* first = &first_arr;
  int* second = &second_arr;
  int* res = &res_arr;
  size_t pi = 0, pj = 0, pos = 0;
  while (pi != first_size && pj != second_size) {
    if (first[pi] < second[pj])
      pi++;
    else if (first[pi] > second[pj])
      pj++;
    else {
      res[pos++] = first[pi];
      pi++;
      pj++;
    }
  }
  return pos;
}

inline size_t SetDifference(int& first_arr, size_t first_size, int& second_arr, size_t second_size, int& res_arr){
  int* first = &first_arr;
  int* second = &second_arr;
  int* res = &res_arr;
  size_t pi = 0, pj = 0, pos = 0;
  while (pi != first_size && pj!=second_size){
      int left = first[pi]; 
      int right = second[pj];
      if(left<=right) pi++;
      if(right<=left) pj++;
      
      if (left < right) {
        res[pos++] = left;
      }
  }
  while(pi<first_size){
    int left = first[pi++];
        res[pos++]=left;
  }
  return pos;
}

bool BinarySearch(int& arr, int start, int end, int target){
    int* nums = &arr;

    size_t mid;
    size_t low = start;
    size_t high = end;

    while (low < high) {
        mid = low + (high - low) / 2;
        if (target <= nums[mid]) {
            high = mid;

        }
        else {
            low = mid + 1;
        }
    }
    if(low < end && nums[low] < target) {
       low++;
    }
    if(low>=start && low < end &&  nums[low]==target) return true;
    else return false;
}

RTYPE FNAME(ARGS)
{
    CODE
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &FNAME);
}