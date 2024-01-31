#include <iostream>
using namespace std;
#include <vector>
#include <algorithm>


/*
����������С�ֱ�Ϊ m �� n �����򣨴�С�������� nums1 �� nums2��
�����ҳ���������������������� ��λ�� ��
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

