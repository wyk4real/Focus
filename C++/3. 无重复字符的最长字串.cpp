//#include <iostream>
//using namespace std;
//#include <unordered_set>
//
//
///*
//给定一个字符串 s ，请你找出其中不含有重复字符的 最长子串 的长度。
//
//unordered_set: 无序集合是一种使用哈希表实现的无序关联容器，
//其中键被哈希到哈希表的索引位置，因此插入操作总是随机的。
//*/
//
//
//class Solution {
//public:
//    int lengthOfLongestSubstring(string s) {
//        // 哈希集合，记录每个字符是否出现过
//        unordered_set<char> occ;
//        int n = s.size();
//        // 右指针，初始值为 -1，相当于我们在字符串的左边界的左侧，还没有开始移动
//        int rk = -1, ans = 0;
//        // 枚举左指针的位置，初始值隐性地表示为 -1
//        for (int i = 0; i < n; ++i) {
//            if (i != 0) {
//                // 左指针向右移动一格，移除一个字符
//                occ.erase(s[i - 1]);
//            }
//            while (rk + 1 < n && !occ.count(s[rk + 1])) {
//                // 不断地移动右指针
//                occ.insert(s[rk + 1]);
//                ++rk;
//            }
//            // 第 i 到 rk 个字符是一个极长的无重复字符子串
//            ans = max(ans, rk - i + 1);
//        }
//        return ans;
//    }
//};
//
//
//
//int main()
//{
//    Solution s;
//    string str = "abcabcbb";
//    int length = s.lengthOfLongestSubstring(str);
//    cout << "length = " << length << endl;
//
//
//    system("pause");
//    return 0;
//
//}
//
