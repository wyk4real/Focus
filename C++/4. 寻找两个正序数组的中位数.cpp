#include <iostream>
using namespace std;
#include <vector>
#include <algorithm>


/*
给定两个大小分别为 m 和 n 的正序（从小到大）数组 nums1 和 nums2。
请你找出并返回这两个正序数组的 中位数 。
*/


class Solution {
public:
    double findMedianSortedArrays(vector<int>& nums1, vector<int>& nums2) {

        vector<int> v;
        v.resize(nums1.size() + nums2.size());
        merge(nums1.begin(), nums1.end(), nums2.begin(), nums2.end(), v.begin());

        int length = v.size();
        
        if (length != 0 && length % 2 == 0) {
            
            return (v[length / 2] + v[length / 2 - 1]) / 2.0;

        }
        else
        {
            return v[length / 2];
        }

    }
};


int main()
{
    vector<int> nums1 = { 1,3};
    vector<int> nums2 = { 2 };

    vector<int> nums1 = { 1,2 };
    vector<int> nums2 = { 3,4 };


    Solution s;
    double res = s.findMedianSortedArrays(nums1, nums2);
    cout << "res = " << res << endl;


    system("pause");
    return 0;

}

