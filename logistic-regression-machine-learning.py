import numpy as np

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

class LogisticRegression:
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.weight = np.zeros((X.shape[1]))
        self.bias = np.zeros(1)

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def predictions(self, x):
        z = np.dot(x, self.weight) + self.bias
        out = self.sigmoid(z)
        return out

    def training(self, epoch=20, lr=0.01, details=False, break_train=0.4):
        m = self.X.shape[0]
        
        for _ in range(epoch):
            y_pred = self.predictions(self.X)
            error = self.y - y_pred

            gradient_w = np.dot(self.X.T, error) / m
            gradient_b = np.mean(error)

            self.weight += lr * gradient_w
            self.bias += lr * gradient_b

            loss = np.mean(np.abs(error))
            if details:
                print(f"loss: {loss}")
            if loss < break_train:
                break

def classify_binary(clsf_output, name_of_clsf):
    if clsf_output < 0.5:
        return name_of_clsf[0]
    else:
        return name_of_clsf[1]

x = [
     "Hôm nay tôi cảm thấy rất chán nản và không có động lực để làm bất cứ việc gì.",
     "Công việc hiện tại của tôi không mang lại niềm vui hay sự thỏa mãn nào, mỗi ngày đều là một cuộc chiến với chính bản thân mình.",
     "Mối quan hệ của tôi với gia đình và bạn bè ngày càng xa cách, dường như không ai hiểu tôi cả.",
     "Tôi liên tục gặp thất bại trong các dự án cá nhân, dường như mọi cố gắng của tôi đều vô ích.",
     "Tôi cảm thấy mình đang lạc lối và không biết tương lai sẽ ra sao, mọi thứ đều mờ mịt và bất định.",
     "Sức khỏe của tôi đang ngày càng xấu đi, tôi luôn cảm thấy mệt mỏi và kiệt sức.",
     "Tài chính của tôi đang gặp khó khăn nghiêm trọng, các khoản nợ ngày càng chồng chất và tôi không biết phải làm sao để thoát ra.",
     "Công ty tôi đang làm việc có môi trường làm việc rất tiêu cực, không ai hỗ trợ hay động viên lẫn nhau.",
     "Cuộc sống của tôi dường như chỉ toàn là những chuỗi ngày vô nghĩa và không có điểm nhấn.",
     "Mọi người xung quanh dường như đều thành công và hạnh phúc, trong khi tôi vẫn đứng yên một chỗ, không thể tiến lên.",
     "Tôi cảm thấy buồn bả và cô đơn quá",
     "tôi đang buồn và cảm thấy tự ti",
     "tôi cảm thấy buồn bả tự ti và tiêu cực",
     
     "Hôm nay là một ngày tuyệt vời và tôi cảm thấy tràn đầy năng lượng để hoàn thành mọi nhiệm vụ.",
     "Công việc của tôi mang lại niềm vui và sự thỏa mãn, tôi rất tự hào về những gì mình đã đạt được.",
     "Mối quan hệ của tôi với gia đình và bạn bè ngày càng gắn bó, chúng tôi luôn ủng hộ và hiểu nhau.",
     "Tôi đã đạt được nhiều thành công trong các dự án cá nhân, mọi cố gắng đều được đền đáp xứng đáng.",
     "Tôi cảm thấy tự tin và hướng về một tương lai tươi sáng, mọi thứ đều trở nên rõ ràng và có mục tiêu.",
     "Sức khỏe của tôi đang được cải thiện từng ngày, tôi luôn cảm thấy tràn đầy sức sống và năng lượng.",
     "Tài chính của tôi đang ổn định và phát triển, tôi có đủ khả năng để quản lý và đầu tư một cách thông minh.",
     "Công ty tôi đang làm việc có môi trường làm việc tích cực, mọi người luôn hỗ trợ và động viên lẫn nhau.",
     "Cuộc sống của tôi đầy ý nghĩa và thú vị, mỗi ngày đều là một trải nghiệm mới mẻ và đáng nhớ.",
     "Tôi tự tin với những gì mình đang làm, và tôi tin rằng thành công và hạnh phúc sẽ đến với mình.",
     "tôi cảm thấy vui quá hehe",
     "tôi cảm thấy hạnh phúc tự tin và vui vẻ",
     "tôi đang cảm thấy rất vui vẻ"
     ]
y = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1

]

bag_of_word = BagOfWord(x)
x_bag_of_word = bag_of_word.fit(x)

model = LogisticRegression(x_bag_of_word, y)
model.training(
    epoch=10000,
    lr=0.1,
    details=False,
    break_train=0.0000001
)

x_test_negative = ["tôi cảm thấy mệt mỏi và cực kỳ tự ti về bản thân"]
x_test_positive = ["tôi đang cảm thấy rất vui vẻ và hạnh phúc về bản thân"]

x_test_bag_of_word = bag_of_word.fit(x_test_negative)
output = model.predictions(x_test_bag_of_word)

print(output)
predict_op = classify_binary(output, ["văn bản tiêu cực", "văn bản tích cực"])

print(f"kết quả dự đoán cho câu '{x_test_negative[0]}' là '{predict_op}")
