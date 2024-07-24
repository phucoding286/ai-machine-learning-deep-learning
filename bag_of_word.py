class BagOfWord:
    def __init__(self, batch_data):
        self.batch_data = batch_data
        self.max_len = 0
    
    # hàm đếm số lần xuất hiện của từ
    def count(self, sentence):
        bag = []
        for char in list(sentence): 
            appeer_times = 0
            for char_count in list(sentence): appeer_times += 1 if char in char_count else 0
            bag.append(appeer_times - 1 if appeer_times > 1 else 0)
        return bag
    
    def fit(self, bag_txt):
        # tạo lô mã hóa với cách điếm số lần xuất hiện của từ
        batch_bag = []
        for sentence in bag_txt:
            count_out = self.count(sentence)
            batch_bag.append(count_out)
            if len(count_out) > self.max_len:
                self.max_len = len(count_out)
        # chuẩn hóa độ dài của văn bản mã hóa
        for i in range(len(batch_bag)):
            if len(batch_bag[i]) < self.max_len: 
                batch_bag[i] = batch_bag[i] + [0 for _ in range(len(batch_bag[i]), self.max_len)]
            else:
                batch_bag[i] = batch_bag[i][:self.max_len]

        return np.array(batch_bag)
