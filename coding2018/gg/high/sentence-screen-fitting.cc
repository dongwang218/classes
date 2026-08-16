// build prefix length of sentence, binary search on it.
// but solution is more clever and faster too.
// http://massivealgorithms.blogspot.com/2016/10/leetcode-418-sentence-screen-fitting.html
public class Solution {
    public int wordsTyping(String[] sentence, int rows, int cols) {
        if(sentence == null || sentence.length == 0 || rows<=0 || cols <=0) return 0;
        String s = String.join(" ", sentence) + " ";
        int l = s.length();
        int start = 0;
        for(int i=0; i<rows; i++){
            start += cols;
            if(s.charAt(start % l) == ' '){
                start++;
            } else {
                while(start > 0 && s.charAt((start-1) % l) != ' '){
                    start--;
                }
            }
        }
        return start/l;
    }
