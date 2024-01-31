#include <iostream>
using namespace std;
#include <vector>
#include <algorithm>


/*
����һ���������� nums ��һ������Ŀ��ֵ target��
�����ڸ��������ҳ� ��ΪĿ��ֵ target  ���� ���� ������
���������ǵ������±ꡣ
*/

void printVector(vector<int>& v){
    for (vector<int>::iterator it = v.begin(); it != v.end(); it++){
        cout << *it << " ";
    }
    cout << endl;
}


class Solution {
public:
    vector<int> twoSum(vector<int>& nums, int target) {
        int n = nums.size();
        for (int i = 0; i < n; ++i) {
            for (int j = i + 1; j < n; ++j) {
                if (nums[i] + nums[j] == target) {
                    return { i, j };
                }
            }
        }
        return {};
    }
};


int main()
{
    vector<int> nums;
    nums.push_back(2);
    nums.push_back(7);
    nums.push_back(11);
    nums.push_back(15);
    printVector(nums);

    Solution s;
    vector<int> target = s.twoSum(nums, 9);
    printVector(target);


    system("pause");
	return 0;

}
