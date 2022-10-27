import math
import pandas as pd

class HBOS:
    def __init__(self, bin_info_array=[], nominal_array=[], mode_array = []):
        '''
        bin_info_array:每个特征分箱的数量
        nominal_array:哪些特征需要分箱
        mode_array:每个特征是否构建动态宽度直方图
        histogram_list：存储直方图
        '''
        self.bin_info_array = bin_info_array
        self.nominal_array = nominal_array
        self.histogram_list = []
        self.mode_array = mode_array

    # 初始化参数，循环调用构建直方图
    def fit(self, data):
        # attr_size:维度（特征数）
        # total_data_size:样本数
        attr_size = len(data.columns)
        total_data_size = len(data)

        # 初始化参数
        # 如果没有自定义，默认标准箱子大小为样本数的平方根
        len_bin_info_array = len(self.bin_info_array)
        if len_bin_info_array < attr_size:
            for _ in range(len_bin_info_array,attr_size):
                    self.bin_info_array.append(round(math.sqrt(len(data))))

        # 如果没有自定义，则默认该特征需要构建直方图
        len_nominal_array = len(self.nominal_array)
        if len_nominal_array < attr_size:
            for _ in range(len_nominal_array,attr_size):
                self.nominal_array.append(False)

        # 如果没有自定义，则默认该特征构建动态直方图
        len_mode_array = len(self.mode_array)
        if len_mode_array < attr_size:
            for _ in range(len_mode_array,attr_size):
                self.mode_array.append('dynamic binwidth')

        # 分数标准化参数，标准化为百分比
        normal = 1.0

        #初始化直方图
        self.histogram_list = []
        for i in range(attr_size):
            self.histogram_list.append([])

        #为每个属性保存最大值（需要规范化 _bin width）
        maximum_value_of_rows = data.apply(max).values

        #排序数据
        sorted_data = data.apply(sorted)

        #创建直方图
        for attrIndex in range(len(sorted_data.columns)):
            attr = sorted_data.columns[attrIndex]
            last = 0
            bin_start = sorted_data[attr][0]
            # 若该特征的mode_array是'dynamic binwidth'，则构建动态宽度直方图
            if self.mode_array[attrIndex] == 'dynamic binwidth':
                # 若该特征的nominal_array是‘True’，则不构建动态直方图（这里等效于构建直方图，但每个数据在该特征的直方图的高度都为1）
                if self.nominal_array[attrIndex]:
                    while last < len(sorted_data) - 1:
                        last = self.create_dynamic_histogram(self.histogram_list, sorted_data, last, 1, attrIndex, True)
                else:
                # 若该特征的nominal_array是‘False’，则构建动态宽度直方图
                    length = len(sorted_data)
                    binwidth = self.bin_info_array[attrIndex]
                    while last < len(sorted_data) - 1:
                        values_per_bin = math.floor(len(sorted_data) / self.bin_info_array[attrIndex])
                        last = self.create_dynamic_histogram(self.histogram_list, sorted_data, last, values_per_bin,
                                                             attrIndex, False)
                        if binwidth > 1:
                            length = length - self.histogram_list[attrIndex][-1].quantity
                            binwidth = binwidth - 1

            # 若该特征的mode_array不是'dynamic binwidth'，则构建静态宽度直方图
            else:
                count_bins = 0
                # 静态直方图的宽度固定
                binwidth = (sorted_data[attr][len(sorted_data) - 1] - sorted_data[attr][0]) * 1.0 / self.bin_info_array[
                    attrIndex]
                if (self.nominal_array[attrIndex]) | (binwidth == 0):
                    binwidth = 1
                while last < len(sorted_data):
                    is_last_bin = count_bins == self.bin_info_array[attrIndex] - 1
                    last = self.create_static_histogram(self.histogram_list, sorted_data, last, binwidth, attrIndex,
                                                        bin_start, is_last_bin)
                    bin_start = bin_start + binwidth
                    count_bins = count_bins + 1

        # 计算保存每个特征的最大分数（高度），并标准化分数
        max_score = []

        # 所有直方图的循环，计算每个特征的最大分数
        for i in range(len(self.histogram_list)):
            max_score.append(0)
            histogram = self.histogram_list[i]

            # 所有箱子的bins
            for k in range(len(histogram)):
                _bin = histogram[k]
                _bin.total_data_size = total_data_size
                _bin.calc_score(maximum_value_of_rows[i])
                if max_score[i] < _bin.score:
                    max_score[i] = _bin.score

        # 标准化分数
        for i in range(len(self.histogram_list)):
            histogram = self.histogram_list[i]
            for k in range(len(histogram)):
                _bin = histogram[k]
                _bin.normalize_score(normal, max_score[i])

    # 计算每个数据的HBOS值（通过该数据的每个特征的分数得到）
    def predict(self, data):
        score_array = []
        for i in range(len(data)):
            each_data = data.values[i]
            value = 1
            for attr in range(len(data.columns)):
                score = self.get_score(self.histogram_list[attr], each_data[attr])
                value = value * score
            score_array.append(value)
        return score_array

    # 返回所有数据的HBOS值
    def fit_predict(self, data):
        data = pd.DataFrame(data)
        self.fit(data)
        return self.predict(data)

    # 获取某数据在某特征的分数
    @staticmethod
    def get_score(histogram, value):
        for i in range(len(histogram) - 1):
            _bin = histogram[i]
            if (_bin.range_from <= value) & (value < _bin.range_to):
                return _bin.score

        _bin = histogram[-1]
        if (_bin.range_from <= value) & (value <= _bin.range_to):
            return _bin.score
        return 0

    # 检查给定值的value_per_bin值是否超过数据的个数
    @staticmethod
    def check_amount(sorted_data, first_occurrence, values_per_bin, attr):

        if first_occurrence + values_per_bin < len(sorted_data):
            if sorted_data[attr][first_occurrence] == sorted_data[attr][first_occurrence + values_per_bin]:
                return True
            else:
                return False
        else:
            return False

    #构造动态宽度直方图
    @staticmethod
    def create_dynamic_histogram(histogram_list, sorted_data, first_index, values_per_bin, attr_index, is_nominal):
        attr = sorted_data.columns[attr_index]

        #创建新的_bin
        _bin = HistogramBin(sorted_data[attr][first_index], 0, 0)

        #检查数据是否接近尾端
        if first_index + values_per_bin < len(sorted_data):
            last_index = first_index + values_per_bin
        else:
            last_index = len(sorted_data)

        #第一个值始终指向_bin
        _bin.add_quantitiy(1)

        '''
        对于每一个其他值，检查是否与最后一个值相同
        如果是的话，将其放入_bin
        如果没有，则检查该值的每个_bin是否有多个值
        如果是的话，打开新的_bin
        如果没有，继续将值放入_bin
        '''

        cursor = first_index
        for i in range(int(first_index + 1), int(last_index)):
            if sorted_data[attr][i] == sorted_data[attr][cursor]:
                _bin.add_quantitiy(1)
                cursor = cursor + 1
            else:
                if HBOS.check_amount(sorted_data, i, values_per_bin, attr):
                    break
                else:
                    _bin.add_quantitiy(1)
                    cursor = cursor + 1

        #继续将值放入_bin，直到新值到达
        for i in range(cursor + 1, len(sorted_data)):
            if sorted_data[attr][i] == sorted_data[attr][cursor]:
                _bin.quantity = _bin.quantity + 1
                cursor = cursor + 1
            else:
                break

        #调整bins范围
        if cursor + 1 < len(sorted_data):
            _bin.range_to = sorted_data[attr][cursor + 1]
        else:  #上次数据
            if is_nominal:
                _bin.range_to = sorted_data[attr][len(sorted_data) - 1] + 1
            else:
                _bin.range_to = sorted_data[attr][len(sorted_data) - 1]

        #存储 _bin
        if _bin.range_to - _bin.range_from > 0:
            histogram_list[attr_index].append(_bin)
        elif len(histogram_list[attr_index]) == 0:
            _bin.range_to = _bin.range_to + 1
            histogram_list[attr_index].append(_bin)
        else:
            # 如果_bin的长度为零
            # 我们将其与以前的_bin合并
            # 因为这可能发生在直方图的末尾
            last_bin = histogram_list[attr_index][-1]
            last_bin.add_quantitiy(_bin.quantity)
            last_bin.range_to = _bin.range_to

        return cursor + 1

    # 构建静态直方图分箱，根据参数直方图列表、排序数据、分箱宽度等参数构建
    @staticmethod
    def create_static_histogram(histogram_list, sorted_data, first_index, binwidth, attr_index, bin_start, last_bin):
        attr = sorted_data.columns[attr_index]
        _bin = HistogramBin(bin_start, bin_start + binwidth, 0)
        # 没到最后，继续调用
        if last_bin:
            _bin = HistogramBin(bin_start, sorted_data[attr][len(sorted_data) - 1], 0)

        last = first_index - 1
        cursor = first_index

        while True:
            # 超过样本数量
            if cursor >= len(sorted_data):
                break
            # 超过范围
            if sorted_data[attr][cursor] > _bin.range_to:
                break
            _bin.quantity = _bin.quantity + 1
            last = cursor
            cursor = cursor + 1

        histogram_list[attr_index].append(_bin)
        return last + 1

# 定义直方图分箱的属性和函数

class HistogramBin:
    '''
    属性包括分箱的左范围、右范围、面积，个数、分数
    函数包括计算高度，定义面积，计算分数、标准化分数
    '''
    def __init__(self, range_from, range_to, quantity):
        self.range_from = range_from
        self.range_to = range_to
        self.quantity = quantity
        self.score = 0
        self.total_data_size = 0

    # 计算分箱的高度
    def get_height(self):
        width = self.range_to - self.range_from
        height = self.quantity / width
        return height

    # 面积，对于动态直方图，每个分箱面积固定
    def add_quantitiy(self, anz):
        self.quantity = self.quantity + anz

    # 计算分箱的分数
    def calc_score(self, max_score):
        if max_score == 0:
            max_score = 1

        if self.quantity > 0:
            self.score = self.quantity / (
                (self.range_to - self.range_from) * self.total_data_size / abs(max_score))

    # 标准化分数
    def normalize_score(self, normal, max_score):
        self.score = self.score * normal / max_score
        if self.score == 0:
            return
        self.score = 1 / self.score
