import akshare as ak
import pandas as pd

def add_market_suffix(code):
    """
    根据股票代码添加市场后缀
    6开头 -> .SH (上海)
    0、3开头 -> .SZ (深圳)
    8开头 -> .BJ (北京)
    """
    code_str = str(code).strip()
    if not code_str:
        return code_str
    
    # 如果已经有后缀，直接返回
    if '.' in code_str:
        return code_str
    
    # 确保是6位数字
    if len(code_str) == 6:
        if code_str.startswith('6'):
            return code_str + '.SH'
        elif code_str.startswith(('0', '3')):
            return code_str + '.SZ'
        elif code_str.startswith('8'):
            return code_str + '.BJ'
    
    return code_str

# 方法1：通过行业板块获取（推荐，速度快）
print("方法1：通过行业板块获取食品饮料股票...")
print("正在获取所有行业板块...")
industry_boards = ak.stock_board_industry_name_em()

# 查找食品饮料相关的行业板块
food_related_boards = industry_boards[industry_boards['板块名称'].str.contains('食品|饮料|酒|乳业|调味', na=False)]
print(f"\n找到相关行业板块：")
print(food_related_boards[['板块名称', '板块代码']].to_string(index=False))

# 获取这些板块的所有股票
all_stocks = []
for _, row in food_related_boards.iterrows():
    board_name = row['板块名称']
    board_code = row['板块代码']
    print(f"\n正在获取 {board_name} 的股票列表...")
    try:
        stocks = ak.stock_board_industry_cons_em(symbol=board_name)
        if stocks is not None and len(stocks) > 0:
            # 确保有代码和名称列
            if '代码' in stocks.columns and '名称' in stocks.columns:
                all_stocks.append(stocks[['代码', '名称']])
            elif '股票代码' in stocks.columns and '股票名称' in stocks.columns:
                stocks = stocks.rename(columns={'股票代码': '代码', '股票名称': '名称'})
                all_stocks.append(stocks[['代码', '名称']])
    except Exception as e:
        print(f"  获取 {board_name} 失败: {e}")
        continue

if all_stocks:
    # 合并所有股票，去重
    result_df = pd.concat(all_stocks, ignore_index=True)
    result_df = result_df.drop_duplicates(subset=['代码'], keep='first')
    result_df = result_df.rename(columns={'代码': 'code', '名称': 'name'})
    # 添加市场后缀
    result_df['code'] = result_df['code'].apply(add_market_suffix)
    result_df = result_df.sort_values(by='code')
    
    print(f"\n找到 {len(result_df)} 只食品饮料相关股票：\n")
    print(result_df.to_string(index=False))
    
    # 保存到文件
    output_file = 'food_beverage_stocks.csv'
    result_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\n结果已保存到: {output_file}")
else:
    print("\n方法1未获取到数据，尝试方法2...")
    
    # 方法2：逐个查询股票行业信息（较慢但准确）
    print("\n方法2：逐个查询股票行业信息...")
    print("正在获取全市场股票代码和名称...")
    all_stocks_df = ak.stock_info_a_code_name()
    
    food_beverage_stocks = []
    total = len(all_stocks_df)
    
    for idx, row in all_stocks_df.iterrows():
        code = row['code']
        name = row['name']
        if (idx + 1) % 100 == 0:
            print(f"进度: {idx + 1}/{total}")
        
        try:
            info = ak.stock_individual_info_em(symbol=code)
            if info is not None and len(info) > 0:
                # 查找行业信息
                industry_row = info[info['item'] == '行业']
                if len(industry_row) > 0:
                    industry = industry_row.iloc[0]['value']
                    if pd.notna(industry) and ('食品' in str(industry) or '饮料' in str(industry) or '酒' in str(industry)):
                        # 添加市场后缀
                        code_with_suffix = add_market_suffix(code)
                        food_beverage_stocks.append({'code': code_with_suffix, 'name': name, 'industry': industry})
        except Exception as e:
            continue
    
    if food_beverage_stocks:
        result_df = pd.DataFrame(food_beverage_stocks)
        result_df = result_df.sort_values(by='code')
        print(f"\n找到 {len(result_df)} 只食品饮料相关股票：\n")
        print(result_df[['code', 'name']].to_string(index=False))
        
        # 保存到文件
        output_file = 'food_beverage_stocks.csv'
        result_df[['code', 'name']].to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"\n结果已保存到: {output_file}")
    else:
        print("\n未找到食品饮料相关股票")

