import random as rd

def lottery_game(total_round=10000):
    '''
    有三张彩票 只有一张中奖你买走一张然后老板当场开了一张没中
    给你个机会：你可以用剩下的一张换你手里的 换不换？
    求概率
    :return: 返回我买的彩票中奖的概率和交换后中将的概率
    '''
    #我抽到奖的次数
    cnt_my_win = 0
    #交换抽到奖的次数
    cnt_cg_win = 0
    for _ in range(total_round):
        lottorys = get_rand_lotterys()
        my_lottery = get_my_lottery()
        store_keeper_lottery = get_store_keeper_lottery(lottorys, my_lottery)
        #我中奖
        if lottorys[my_lottery] == 1:
            cnt_my_win += 1
        #我交换
        else:
            cnt_cg_win += 1
    return cnt_my_win / total_round, cnt_cg_win / total_round

def get_rand_lotterys(size=3):
    '''
    :param size: 彩票张数
    :return: 一个随机生成的彩票数组，其中只有一张中奖
    '''
    lotterys = [0] * size
    rand_num = rd.randint(0, size-1)
    lotterys[rand_num] = 1
    return lotterys

def get_my_lottery(size=3):
    '''
    我随机选一张彩票
    :param size: 彩票张数
    :return:
    '''
    return rd.randint(0, size-1)

def get_store_keeper_lottery(lotterys, my_lottery, size=3):
    '''
    获取商店老板得到的没有中将的彩票
    :return:
    '''
    while (True):
        rand_num = rd.randint(0, size-1)
        #是中将的跳过
        if lotterys[rand_num] == 1: continue
        #我选中的跳过
        elif rand_num == my_lottery: continue

        return rand_num

if __name__ == '__main__':
    my_win_rate, cg_win_rate = lottery_game()
    print('我中奖的概率: ', my_win_rate, ' 交换中奖的概率: ', cg_win_rate)