
#please pay attention to all comments are written in chinese!
from DrissionPage import ChromiumOptions, ChromiumPage
import time
import pandas as pd

# Set the browser path
path = r'C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe'
co = ChromiumOptions().set_browser_path(path)

# open the browser
page = ChromiumPage(co)

# Monitoring data packet characteristics
page.listen.start('https://api.m.jd.com/?appid=item-v3&functionId=pc_club_productPageComments')

# visit website
page.get('https://item.jd.com/100005485240.html')

# wait
page.wait.load_start()

# Click the "View Comment Data" button
try:
    page.ele('css:#detail > div.tab-main.large > ul > li.current').click()
except Exception as e:
    print(f'点击评论按钮失败: {e}')
    page.quit()
    exit()

# Crawl multiple pages of comments
all_comments = []
target_comments = 980  # 目标评论数
page_num = 1

while len(all_comments) < target_comments:
    print(f'正在爬取第 {page_num} 页评论...')
    try:
        resp = page.listen.wait()
        json_data = resp.response.body
        comments = json_data['comments']

        # 只添加评分<=4的评论
        filtered_comments = [comment for comment in comments if comment.get('score', 5) <= 4]
        all_comments.extend(filtered_comments)

        if len(all_comments) >= target_comments:
            break

        # 点击"下一页"按钮
        try:
            next_page_btn = page.ele(
                'css:#comment-0 > div.com-table-footer > div > div > a.ui-pager-next')  # 替换为实际的"下一页"按钮选择器
            next_page_btn.click()
            time.sleep(2)  # 等待页面加载
            page_num += 1
        except Exception as e:
            print(f'点击"下一页"按钮失败: {e}')
            break
    except Exception as e:
        print(f'获取评论数据失败: {e}')
        break

# Convert the comment data into a DataFrame (only retain comments with a rating of <=4)
comments_data = []
for comment in all_comments:
    comments_data.append({
        '用户昵称': comment.get('nickname', '未知用户'),  # 用户昵称
        '评论内容': comment.get('content', ''),  # 评论内容
        '评分': comment.get('score', 0),  # 评分
        '评论时间': comment.get('creationTime', '')  # 评论时间
    })

df = pd.DataFrame(comments_data)

# Save as an Excel file
excel_file = 'comments2.xlsx'
df.to_excel(excel_file, index=False, engine='openpyxl')

print(f'已保存 {len(all_comments)} 条评论到 {excel_file}')

# lsiten stop
page.listen.stop()

# close the brouser
page.quit()
