import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
import joblib

st.set_page_config(
    page_title="期末考试", # 页面标题
    page_icon="💯", # 页面图标
    layout="wide", # 布局方式
)

model = joblib.load("score_predictor.pkl") # 加载训练好的模型
# -------------------- 项目介绍页 --------------------
def xiangmu_page():
    st.markdown('# 🎓学生成绩分析与预测系统')
    s1, s2 = st.columns([3,2])
    with s1:   
        st.markdown('## 📄项目概述:')
        st.text('本项目是一个基于Streamlit的学生成绩分析平台，通过数据可视化和机器学习技术，帮助教育工作者和学生深入了解学业表现，并预测期末考试成绩。')

        st.markdown('## ⭐主要特点:')
        st.text(
            """
            ▪📊 数据可视化：多维度展示学生学业数据
            ▪🎯 专业分析：按专业分类的详细统计分析
            ▪🔮 智能预测：基于机器学习模型的成绩预测
            ▪💡 学习建议：根据预测结果提供个性化反馈
            """
            )
    with s2:
        images=[
        {
          'url':'images/1.jpg',
          'parm':'各专业男女性别比例'
        },
        {
          'url':'images/2.jpg',
          'parm':'各专业学习指标对比'
        },
        {
          'url':'images/3.jpg',
          'parm':'成绩待提高,建议加强学习'
        },
        {
          'url':'images/4.jpg',
          'parm':'成绩合格，继续保持'
        }
        ]

        st.subheader('✅预览')
        if 'ind' not in st.session_state:
           st.session_state['ind']= 0

        def nextImg ():
            st.session_state['ind'] = (st.session_state['ind'] - 1) % len(images)
        def nextWmg ():
            st.session_state['ind'] = (st.session_state['ind'] + 1) % len(images)
            
        st.image(images[st.session_state['ind']]['url'],caption=images[st.session_state['ind']]['parm'])

        c1,c2=st.columns(2)
        with c1:
            st.button('上一张',on_click=nextImg,use_container_width=True)
        with c2:
            st.button('下一张',on_click=nextWmg,use_container_width=True)





    st.markdown('## 🚀项目目标')
    a1, a2, a3 = st.columns([2,2,2])
    with a1:
       st.markdown('### 🎯目标一')
       st.text('分析影响因素')
       st.text(
        """
        ▪ 识别关键学习指标
        ▪ 探索成绩相关因素
        ▪ 提供数据支持决策
        """
        )
    with a2:
       st.markdown('### 🎯目标二')
       st.text('可视化展示')
       st.text(
        """
        ▪ 专业对比分析
        ▪ 性别差异研究
        ▪ 学习模式识别
        """
        )
    with a3:
        st.markdown('### 🎯目标三')
        st.text('成绩预测')
        st.text(
        """
        ▪ 机器学习模型
        ▪ 个性化预测
        ▪ 及时干预预警
        """
        )

    st.markdown('## 🛠技术架构')
    c1, c2, c3, c4 = st.columns([2,2,2,2])
    with c1:
        st.text('前端框架')
        python_code ='streamlit'
        st.code(python_code, language=None)
    with c2:
        st.text('数据处理')
        python_code ='''
        pandas 
        Numpy
                 '''
        st.code(python_code, language=None)
    with c3:
        st.text('可视化')
        python_code ='''
        plotly 
        Natplotlib
                '''
        st.code(python_code, language=None)
    with c4:
        st.text('机器学习')
        python_code ='scikit-learn'
        st.code(python_code, language=None)
    
# -------------------- 专业数据分析页 --------------------
def zhuanye_page():
    st.markdown('# 🎓专业数据分析')
    df = pd.read_csv('D:\streamlit_env\student_data_adjusted_rounded.csv')
    st.subheader("1. 各专业男女性别比例")
    col1, col2 = st.columns([2, 1])
    with col1:
        gender_fig = px.histogram(df, x="专业", color="性别", barmode="group",
                                  title="各专业男女性别比例",
                                  labels={"专业": "专业", "count": "人数"})
        st.plotly_chart(gender_fig, use_container_width=True)
    st.markdown("---")
    st.subheader("2. 各专业学习指标对比")
    with col2:
        gender_data = df.groupby(["专业", "性别"])["学号"].count().reset_index()
        gender_pivot = gender_data.pivot(index="专业", columns="性别", values="学号").fillna(0)
        st.subheader("性别比例数据")
        st.dataframe(gender_pivot, use_container_width=True)
    

    metrics = ["每周学习时长（小时）", "期中考试分数", "期末考试分数"]
    metric_df = df.groupby("专业")[metrics].mean().reset_index()
    col1, col2 = st.columns([2, 1])

    with col1:
        metric_fig = go.Figure()
        # 单独处理“每周学习时长”为柱状图
        metric_fig.add_trace(
            go.Bar( # 改为Bar类型（柱状图）
                x=metric_df["专业"], 
                y=metric_df["每周学习时长（小时）"], 
                name="每周学习时长（小时）"
            )
        )
        # 其他指标保留折线图
        for metric in metrics[1:]: # 从第2个指标开始遍历
            metric_fig.add_trace(
                go.Scatter(
                    x=metric_df["专业"], 
                    y=metric_df[metric], 
                    name=metric
                )
                )
        # 更新图表布局
        metric_fig.update_layout(
            title="各专业学习指标趋势对比",
            xaxis_title="专业", 
            yaxis_title="指标值"
        )
        st.plotly_chart(metric_fig, use_container_width=True)

    with col2:
        st.subheader("详细数据")
        st.dataframe(metric_df, use_container_width=True)
    
    # 分割线       
    st.markdown("---")
    st.subheader("3. 各专业出勤率分析")
    col1, col2 = st.columns([2, 1])

    # 先计算各专业的平均出勤率
    df_avg_attendance = df.groupby("专业")["上课出勤率"].mean().reset_index() # 按专业分组求平均

    with col1:
        # 热力图：基于平均出勤率数据
        attendance_fig = px.density_heatmap(
            df_avg_attendance, # 改用分组后的平均数据
            x="专业", 
            y="上课出勤率",
            title="各专业出勤率分布（平均值）", # 标题说明是平均值
            labels={"专业": "专业", "上课出勤率": "平均上课出勤率"} # 标签改为“平均”
        )
        st.plotly_chart(attendance_fig, use_container_width=True)

    with col2:
        # 排名表：基于平均出勤率排序
        attendance_rank = df_avg_attendance.sort_values("上课出勤率", ascending=False)
        attendance_rank["排名"] = attendance_rank["上课出勤率"].rank(ascending=False).astype(int) # 计算排名
        st.subheader("出勤率排名（平均值）")
        st.dataframe(attendance_rank[["专业", "上课出勤率"]], use_container_width=True)

        
    # 分割线
    st.markdown("---")
    st.subheader("4. 大数据管理专业专项分析")
    bd_major = df[df["专业"] == "大数据管理"]
    # 关键指标卡片
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("平均出勤率", f"{bd_major['上课出勤率'].mean():.1%}")
    with col2:
        st.metric("期中成绩", f"{bd_major['期中考试分数'].mean():.1f}分")
    with col3:
        st.metric("期末成绩", f"{bd_major['期末考试分数'].mean():.1f}分")
    with col4:
        st.metric("平均学习时长", f"{bd_major['每周学习时长（小时）'].mean():.1f}小时")
    # 图表
    col1, col2 = st.columns(2)
    with col1:
        bd_score_fig = px.histogram(bd_major, x="期末考试分数", title="大数据管理专业期末成绩分布")
        st.plotly_chart(bd_score_fig, use_container_width=True)
    with col2:
        bd_hours_fig = px.box(bd_major, y="每周学习时长（小时）", title="大数据管理专业学习时长分布")
        st.plotly_chart(bd_hours_fig, use_container_width=True)
        
# -------------------- 期末成绩预测页 --------------------
def chengji_page():
    st.markdown('# 🎓期末成绩预测')
    st.subheader("请输入学生的学习信息,系统将预测其期末成绩并提供学习建议")
    q1, q2 = st.columns([2,2])    
    with q1:        
        number = st.text_input('学号', autocomplete='number')
        st.write('性别')
        sex = st.selectbox('性别',['男', '女'],label_visibility='collapsed')
        st.write('专业')
        zhuanye = st.selectbox('专业',['人工智能','大数据管理', '工商管理', '电子商务','财务管理'],label_visibility='collapsed')
        submit = st.button("预测成绩")
    with q2:
        xuexi = st.slider('每周学习时长(小时)', min_value=0.0, max_value=100.0, step=0.1)
        shangke = st.slider('上课出勤率', min_value=0.0, max_value=1.0, step=0.01)
        qizhong= st.slider('期中考试分数', min_value=0.0, max_value=100.0, step=0.1)
        zuoye= st.slider('作业完成率',min_value=0.0, max_value=1.0, step=0.01)
        
    if submit:  
        # 构造输入特征
        X = [[xuexi,shangke,qizhong,zuoye]]
        pred_score = model.predict(X)[0]
        pred_score = max(0, min(100, pred_score)) # 限制分数在0-100之间
        st.subheader("📊预测结果")
        st.markdown(f"**预测期末成绩：{pred_score:.2f} 分**")
        if pred_score >= 80:
            st.image("https://ts1.tc.mm.bing.net/th/id/OIP-C.AaPtSI5bw7iOvdyH5ejhkQHaEn?rs=1&pid=ImgDetMain&o=7&rm=3") # 需提前准备高分反馈图片，替换为实际路径
            st.text('🎉恭喜你!成绩优秀!')
        elif pred_score >= 60:
            st.success("成绩合格，继续保持！")
            st.image("https://img95.699pic.com/element/40263/9350.png_300.png ")
        else:
            st.warning("成绩待提高，建议加强学习！")
            st.image("https://pic.mksucai.com/00/12/17/13961ccd892db3e2.webp ")
       
   
# 在左侧添加侧边栏并设置单选按钮
nav = st.sidebar.radio("🎓导航菜单", ["项目介绍", "专业数据分析","成绩预测"])
# 根据选择的结果，展示不同的页面
if nav == "项目介绍":
    xiangmu_page()
elif nav == "专业数据分析":
    zhuanye_page()
elif nav == "成绩预测":
    chengji_page()



