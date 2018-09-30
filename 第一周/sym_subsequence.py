'''
求字符串最长对称子序列
'''
def symmetric_subsequence(word):
    str_len = len(word)
    subsequence_max = 0
    for i in range(str_len):
    	for j in range(-1,-(str_len+1),-1):
            if (word[i]==word[j]) and i<(j+str_len) :
                count = 1
                subsequence = 0
                while (word[i+count]==word[j-count]) and ((i+count)<(j-count+str_len)):
                    if (i+count+1)==(j-count+str_len):
                        subsequence = count + 1
                    count += 1
                if subsequence>subsequence_max :
                    subsequence_max = subsequence
    print(subsequence_max*2)

symmetric_subsequence('google')
