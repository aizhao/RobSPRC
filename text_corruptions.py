import random
import string

class TextCorruptor:
    def __init__(self):
        # 维基百科常见拼写错误字典 (简易版，可自行扩充)
        self.misspell_map = {
            'their': 'there', 'there': 'their', 'your': 'youre', 
            'definitely': 'definately', 'separate': 'seperate',
            'a lot': 'alot', 'receive': 'recieve', 'until': 'untill'
        }
        # 维基百科常见同音词字典 (简易版，可自行扩充)
        self.homophone_map = {
            'in': 'inn', 'to': 'too', 'write': 'right', 'see': 'sea', 
            'one': 'won', 'two': 'too', 'for': 'four', 'sun': 'son',
            'hair': 'hare', 'night': 'knight', 'flower': 'flour'
        }
        self.qwerty_map = {
            'a': 's', 's': 'd', 'd': 'f', 'o': 'p', 'i': 'o', 
            'e': 'r', 'r': 't', 't': 'y', 'y': 'u', 'u': 'i',
            'w': 'e', 'q': 'w', 'h': 'j', 'j': 'k', 'k': 'l'
        }

    def apply(self, text: str, c_type: str) -> str:
        if not text or c_type in ['none', 'clean']:
            return text
            
        words = text.split()
        if len(words) == 0:
            return text

        c_type = c_type.lower()
        
        # 1. Swap: 随机打乱一个单词内部的两个字符的位置 [cite: 345]
        if c_type == 'swap':
            idx = random.randint(0, len(words) - 1)
            word = list(words[idx])
            if len(word) > 1:
                c_idx = random.randint(0, len(word) - 2)
                word[c_idx], word[c_idx+1] = word[c_idx+1], word[c_idx]
                words[idx] = "".join(word)
                
        # 2. Qwerty: 模拟在QWERTY键盘上按偏，替换为相邻字符 [cite: 346]
        elif c_type == 'qwerty':
            idx = random.randint(0, len(words) - 1)
            word = list(words[idx])
            if len(word) > 0:
                c_idx = random.randint(0, len(word) - 1)
                char = word[c_idx].lower()
                if char in self.qwerty_map:
                    # 保持原有的大小写
                    new_char = self.qwerty_map[char].upper() if word[c_idx].isupper() else self.qwerty_map[char]
                    word[c_idx] = new_char
                words[idx] = "".join(word)

        # 3. RemoveChar: 随机删除单词中的某些字符 [cite: 347]
        elif c_type == 'removechar':
            idx = random.randint(0, len(words) - 1)
            word = words[idx]
            if len(word) > 1:
                c_idx = random.randint(0, len(word) - 1)
                words[idx] = word[:c_idx] + word[c_idx+1:]

        # 4. RemoveSpace: 删除一个空格，合并两个单词 [cite: 348]
        elif c_type == 'removespace':
            if len(words) > 1:
                idx = random.randint(0, len(words) - 2)
                words[idx] = words[idx] + words[idx+1]
                del words[idx+1]

        # 5. Misspelling: 替换为维基百科列表中的常见错误拼写 [cite: 349]
        elif c_type == 'misspelling':
            for i, w in enumerate(words):
                clean_w = w.lower().strip(string.punctuation)
                if clean_w in self.misspell_map:
                    words[i] = w.lower().replace(clean_w, self.misspell_map[clean_w])
                    break

        # 6. Repetition: 随机重复单词 [cite: 350]
        elif c_type == 'repetition':
            idx = random.randint(0, len(words) - 1)
            words.insert(idx, words[idx])

        # 7. Homophone: 更改为其同音词 [cite: 351]
        elif c_type == 'homophone':
            for i, w in enumerate(words):
                clean_w = w.lower().strip(string.punctuation)
                if clean_w in self.homophone_map:
                    words[i] = w.lower().replace(clean_w, self.homophone_map[clean_w])
                    break

        return " ".join(words)