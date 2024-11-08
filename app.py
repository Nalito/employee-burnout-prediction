import streamlit as st
import pickle
import pandas as pd
import numpy as np

model = pickle.load(open('svc.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))
age_enc  = pickle.load(open('age_enc.pkl', 'rb'))
cover_up_enc = pickle.load(open('cover_up_enc.pkl', 'rb'))
is_apppreciated_enc = pickle.load(open('is_appreciated_enc.pkl', 'rb'))
leave_day_satisfaction_enc = pickle.load(open('leave_day_satisfaction_enc.pkl', 'rb'))
love_job_enc = pickle.load(open('love_job_enc.pkl', 'rb'))
more_tasks_than_can_handle_enc = pickle.load(open('more_tasks_than_can_handle_enc.pkl', 'rb'))
working_model_enc = pickle.load(open('working_model_enc.pkl', 'rb'))

def preprocess_data(data):
    data = [data]
    data = pd.DataFrame(data, columns=['age', 'working_model', 'outstanding_tasks', 'leave_day_satisfaction', 'is_appreciated', 'love_job', 'work_culture', 'work_communication', 'work_collaboration', 'work_stress', 'cover_up', 'more_tasks_than_can_handle'])

    data['age'] = age_enc.transform(data['age'])
    data['working_model'] = working_model_enc.transform(data['working_model'])
    data['cover_up'] = cover_up_enc.transform(data['cover_up'])
    data['is_appreciated'] = is_apppreciated_enc.transform(data['is_appreciated'])
    data['leave_day_satisfaction'] = leave_day_satisfaction_enc.transform(data['leave_day_satisfaction'])
    data['love_job'] = love_job_enc.transform(data['love_job'])
    data['more_tasks_than_can_handle'] = more_tasks_than_can_handle_enc.transform(data['more_tasks_than_can_handle'])

    data = scaler.transform(data)

    return data

def predict(data):
    data = preprocess_data(data)
    pred = model.predict(data)
    proba = np.max(model.predict_proba(data))
    burnt = ['Based on your responses, it appears that you are experiencing some signs of burnout. It is important to prioritize self-care and consider taking breaks to recharge. Remember, taking time for yourself can ultimately improve your productivity and well-being. You have come this far—keep going, but make sure to take care of yourself along the way!', "I know work has been intense, and it's completely okay to feel stretched. Remember, your well-being is as important as any project deadline. Taking breaks and setting boundaries helps you recharge, and we're here to support you in finding balance. Let's talk about how we can make things easier for you.", "Your hard work doesn't go unnoticed, and we're grateful to have someone as dedicated as you. Sometimes, the best thing you can do is step back and give yourself a break. We’ll help prioritize what’s essential and find ways to ease some of the load.", "Thank you for everything you’re doing—it’s clear that you’re giving 100%. If you're feeling drained, know that taking time to recover is not just okay; it's essential. Let’s look at where we can adjust things so you have the space you need to come back feeling strong."]
    not_burnt = ['Great news! It seems like you are managing your workload and stress levels effectively. Keep up the good balance, and continue to stay mindful of your limits to prevent burnout. Maintaining this healthy approach will serve you well in the long run!', "Your energy and commitment really shine through! To keep feeling this way, remember to pace yourself and celebrate each small win. Balance now will help keep burnout at bay, and we’re here to help you maintain that positive momentum.", "It's fantastic to see how engaged and driven you are in your work. If you ever start to feel like things are getting intense, don’t hesitate to take a breather or reach out. Staying ahead of burnout is key, and we’re always here to support your well-being.", "We truly appreciate your enthusiasm and the balance you bring to your work. To keep that going, it's great to take regular breaks and avoid overloading yourself. Just remember, we’re here if you ever feel you need a hand managing the workload."]

    if pred[0] == 0:
        recommendation = np.random.choice(not_burnt)
        pred = 'not burnt out'
    else:
        recommendation = np.random.choice(burnt)
        pred = 'burnt out'
    return [pred, proba, recommendation]

def main():
    # Title for the app
    st.title("Employee Burnout Prediction App")

    # Age Range
    age = st.selectbox(
        "What is your age range?",
        ["16-25", "26-40", "41-55", "55+"]
    )

    # Working Model
    working_model = st.selectbox(
        "What working model is employed by your place of work?",
        ["Hybrid", "On-site", "Remote"]
    )

    # Outstanding Work Tasks
    outstanding_tasks = st.number_input(
        "As at the time of filling this form, how many outstanding work tasks do you have?",
        min_value=0, step=1
    )

    # Satisfaction with Annual Leave Days
    leave_day_satisfaction = st.selectbox(
        "Are you satisfied with the number of annual leave days that your company gives you?",
        ["Yes", "No", "Maybe"]
    )

    # Feel Appreciated at Work
    is_appreciated = st.selectbox(
        "Do you feel appreciated at work?",
        ["Yes", "No", "Maybe"]
    )

    # Love Job
    love_job = st.selectbox(
        "Do you love your job?",
        ["Yes", "No", "Maybe"]
    )

    # Work-Induced Stress Level
    work_stress = st.slider(
        "What is your current work-induced stress level?",
        min_value=1, max_value=10, value=5
    )

    # Team Collaboration Level
    work_collaboration = st.slider(
        "How would you rate the level of collaboration within your team?",
        min_value=1, max_value=10, value=5
    )

    # Communication with Manager/Employer/Client
    work_communication = st.slider(
        "How would you rate the level of communication between you and your manager/employer/client?",
        min_value=1, max_value=10, value=5
    )

    # Work Culture Rating
    work_culture = st.slider(
        "How would you rate the work culture of your place of work?",
        min_value=1, max_value=10, value=5
    )

    # Covering for Colleagues
    cover_up = st.selectbox(
        "How often do you have to cover for your colleagues?",
        ["Very often", "Often", "Seldom", "Never"]
    )

    # Taking on More Tasks
    more_tasks = st.selectbox(
        "Do you often take on more tasks than you can handle at work?",
        ["Yes", "No"]
    )

    data = [age, working_model, outstanding_tasks, leave_day_satisfaction, is_appreciated, love_job, work_culture, work_communication, work_collaboration,  work_stress, cover_up, more_tasks]
    

    # Submit Button
    if st.button("Predict Burnout"):
        prediction = predict(data)
        st.write('You are ', prediction[0], '. The model is ', np.round(prediction[1]*100, 2), '% confident of this prediction.')
        st.write(prediction[2])

# Run the app
if __name__ == '__main__':
    main()
