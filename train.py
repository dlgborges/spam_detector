import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GlobalAveragePooling1D, Dense, Dropout

MODEL_PATH = "spam_model.h5"
TOKENIZER_PATH = "tokenizer_config.json"
MAX_WORDS = 10000
MAX_LEN = 100
EMBEDDING_DIM = 16
EPOCHS = 30

# Dataset embutido: mensagens de spam e ham (não-spam)
spam_messages = [
    "Congratulations! You've won a $1000 Walmart gift card. Click here to claim now.",
    "URGENT: Your account has been compromised. Click the link to secure your account immediately.",
    "You have been selected for a free iPhone! Click here to claim your prize.",
    "Make money fast! Work from home and earn $5000 per week. No experience needed.",
    "FREE entry to win a brand new car! Text WIN to 12345 now!",
    "Lose 30 pounds in 30 days! Amazing weight loss secret revealed!",
    "Cheap medications online! Viagra, Cialis, and more at 90% discount!",
    "You are our lucky winner! Claim your million dollar prize now!",
    "Hot singles in your area want to meet you! Click here for free access.",
    "Act now! Limited time offer: Get rich quick with this one secret investment trick.",
    "Nigerian prince needs your help transferring $10 million. You will be rewarded.",
    "FREE FREE FREE! Click here for absolutely free products and gift cards!",
    "Your computer has a virus! Call 1-800 number immediately for tech support.",
    "Congratulations! You have been approved for a pre-paid credit card with $5000 limit.",
    "Earn thousands of dollars weekly from home! Guaranteed income, no skills required.",
    "URGENT: Your bank account will be suspended. Verify your details immediately.",
    "Buy cheap luxury watches and designer brands at 90% off! Limited stock!",
    "You have inherited $2,500,000 from a distant relative. Send your details to claim.",
    "Double your Bitcoin overnight! Guaranteed returns! Invest now!",
    "FREE trial! No credit card needed! Just pay shipping and handling for amazing products.",
    "Exclusive deal just for you! Massive discounts on everything. Don't miss out!",
    "Click here to see who has been viewing your profile. This is your last chance!",
    "Your package could not be delivered. Click here to reschedule delivery immediately.",
    "You have won a free vacation to the Bahamas! Call now to book your trip!",
    "Secret method to make $10,000 per month working only 1 hour per day!",
    "ATTENTION: Unauthorized login attempt detected. Click to secure your account.",
    "Lose weight without diet or exercise! Miracle pill melts fat away!",
    "Get a loan approved in minutes! No credit check! Low interest rates!",
    "Your email has been randomly selected for a $1000 gift card reward!",
    "This is not a scam! Make real money with our proven system. Join today!",
    "You have 24 hours to claim your free gift card before it expires!",
    "Amazing anti-aging cream that erases wrinkles overnight! Order now!",
    "Congratulations! You've been pre-approved for a $50,000 credit line!",
    "Send this message to 10 friends and receive $5000 tomorrow!",
    "Enlarge your business naturally! Thousands of satisfied customers!",
    "You have an outstanding warrant for your arrest! Call immediately to resolve.",
    "Click below to accept your inheritance worth $3.5 million dollars.",
    "Work from home and make $200/hour! No experience! Start today!",
    "URGENT: IRS lawsuit filed against you. Call now to avoid prosecution.",
    "Free ringtones and wallpapers! Download now for your phone!",
    "Miracle supplement cures diabetes, cancer, and heart disease! Buy now!",
    "Your email address has been randomly selected. You've WON $1,000,000!",
    "Double your money in 24 hours! 100% guaranteed! No risk involved!",
    "Last chance! Your account will be permanently deleted in 24 hours!",
    "You have 15 unread premium messages! Click to view attractive singles!",
    "Government grant money available! You qualify for $25,000 free cash!",
    "Act now for a FREE trial of our exclusive product. Pay only shipping!",
    "Amazing opportunity! Invest $100 and receive $5000 in return!",
    "Your prescription is ready. Order cheap drugs with overnight delivery!",
    "You have been chosen for an exclusive cash reward program!",
    "Secret revealed: How I made $50,000 in one week from home!",
    "Congratulations! You just won an all-expenses-paid luxury cruise!",
    "URGENT: PayPal account restricted. Verify identity by clicking here.",
    "Amazing teeth whitening formula! Celebrity secret revealed! Order now!",
    "You are pre-approved for a zero-interest credit card with $10,000 limit!",
    "Get a college degree in days! No studying required! Accredited degrees!",
]

ham_messages = [
    "Hey, are we still meeting for lunch tomorrow at noon?",
    "Can you send me the report by end of day Friday?",
    "Thanks for the update on the project. Everything looks good.",
    "Mom, I'll be home for dinner around 7pm. See you then!",
    "The meeting has been rescheduled to Wednesday at 3pm.",
    "Please review the attached document and let me know your feedback.",
    "Happy birthday! Hope you have an amazing day!",
    "I'll pick up the groceries on my way home from work.",
    "The team did a great job on the presentation today.",
    "Can you call me when you get a chance? Need to discuss something.",
    "Just wanted to say thanks for all your help this week.",
    "The flight departs at 6am tomorrow. Don't forget to check in online.",
    "I found a great restaurant for our anniversary dinner.",
    "Reminder: dentist appointment on Thursday at 2pm.",
    "The kids have soccer practice every Tuesday and Thursday.",
    "Let me know if you need anything from the store.",
    "Great news! The client approved our proposal!",
    "Could you please review my code and give me some feedback?",
    "I'm running a few minutes late. See you in the lobby.",
    "Don't forget to submit your timesheet by end of week.",
    "The weather looks great for our hiking trip this weekend.",
    "Attached are the photos from last night's event.",
    "Hey, did you watch the game last night? Incredible finish!",
    "I'll have the budget report ready by Monday morning.",
    "Thanks for the heads up about the schedule change.",
    "Let's grab coffee sometime this week and catch up.",
    "The new software update fixed the bug we were experiencing.",
    "Please find the invoice attached for your records.",
    "I'm looking forward to the conference next month!",
    "Can you recommend a good book on machine learning?",
    "The delivery will arrive between 2pm and 4pm today.",
    "We should organize a team building event soon.",
    "Thanks for the birthday wishes! It means a lot.",
    "I finished reading that book you recommended. It was fantastic!",
    "Let me confirm our dinner reservation for Saturday.",
    "The project deadline has been extended to next Friday.",
    "Hey, I'm at the office if you need to discuss anything urgent.",
    "Just finished my morning run. Feeling great today!",
    "Can you share the link to the webinar we discussed?",
    "I'll prepare the slides for tomorrow's meeting tonight.",
    "Hope you're having a great vacation! Enjoy!",
    "The IT team will be updating the servers this weekend.",
    "Let me know what time works for the call on Thursday.",
    "Thanks for forwarding the email. I'll take a look.",
    "Our dog had a great time at the pet daycare yesterday.",
    "I submitted the application for the grant. Fingers crossed!",
    "Happy new year! Wishing you all the best for 2024.",
    "The restaurant was excellent. We should go back sometime.",
    "I'm working from home today due to the snowstorm.",
    "Can you proofread my essay before I submit it?",
    "The concert tickets are almost sold out. Want me to buy them now?",
    "I'll be in the conference room if you need to reach me.",
    "Thanks for covering my shift on Friday. I owe you one!",
    "The quarterly results are looking much better than expected.",
    "Let's schedule a follow-up meeting to discuss next steps.",
    "I love the new design for the website. Very clean and modern!",
    "Don't forget we have a doctor's appointment next Monday.",
    "The training session was really helpful. Learned a lot today!",
    "I'll send you the details about the workshop later today.",
    "Great job on completing the project ahead of schedule!",
    "We're planning a barbecue this Sunday. You should come!",
    "The book I ordered arrived today. Can't wait to start reading.",
    "Thanks for the ride to the airport. I really appreciate it.",
    "Our neighbor's cat keeps coming into our yard.",
    "I'll check the availability and get back to you tomorrow.",
    "The recipe turned out great! My family loved it.",
    "Let me know if you want to join us for the movie tonight.",
    "I've been learning Python and it's been really fun!",
    "The office will be closed on Monday for the holiday.",
    "Can you recommend a good mechanic in the area?",
    "Just wrapped up a great workout at the gym.",
    "I forwarded your email to the rest of the team.",
    "The kids had a great time at the birthday party today.",
    "Let me know if you need help with anything this weekend.",
    "The new coffee machine in the office is amazing!",
    "I'll review the contract and send you my comments by Wednesday.",
    "Thanks for the wonderful dinner last night. Everything was delicious!",
    "We need to restock the office supplies. Can you handle that?",
    "Had a productive meeting with the marketing team today.",
    "The gym is offering a special promotion this month.",
    "I'm going to the farmers market this morning. Want to join?",
    "Please remember to water the plants while I'm away.",
    "The webinar covered some really interesting topics about AI.",
    "I'll be traveling next week for the business conference.",
    "Thanks for the great feedback on my presentation!",
    "Our flight was delayed by two hours. Finally arrived home.",
    "The library has a great selection of new books this month.",
    "Can we reschedule our meeting to next Monday?",
    "I just adopted a new puppy! So excited to share photos.",
    "The restaurant recommended by our colleague was fantastic.",
    "I've been meaning to organize my workspace. Today's the day!",
    "The traffic was terrible this morning. Almost late for work.",
    "Let's plan a weekend getaway to the mountains.",
    "The recipe I tried last night was from my grandmother's cookbook.",
    "I'll have the final version of the document by end of day.",
    "Thanks for letting me know about the schedule change.",
    "The neighborhood watch meeting is next Tuesday at 7pm.",
    "I'm really enjoying the new TV series everyone's been talking about.",
    "Our team won the company softball tournament! Celebration tonight!",
    "Can you help me move some furniture this Saturday?",
    "The kids need new school supplies before classes start next week.",
    "I submitted my vacation request. Waiting for approval.",
    "Great weather for a walk in the park this afternoon.",
    "I'll prepare the agenda for tomorrow's team meeting.",
    "Thanks for sharing that interesting article about renewable energy.",
    "The conference call quality was poor today. Hopefully better next time.",
    "We should try that new Italian restaurant downtown.",
    "I'm attending a workshop on data science this weekend.",
    "The car needs an oil change. I'll take it to the shop tomorrow.",
    "Happy Friday! Let's wrap up the week with a team lunch.",
]

def train_and_save_model():
    """Train the spam detection model and save it along with the tokenizer config."""
    
    # Combine data
    all_messages = spam_messages + ham_messages
    labels = [1] * len(spam_messages) + [0] * len(ham_messages)
    
    # Shuffle data
    combined = list(zip(all_messages, labels))
    np.random.seed(42)
    np.random.shuffle(combined)
    all_messages, labels = zip(*combined)
    all_messages = list(all_messages)
    labels = np.array(labels)
    
    # Tokenize
    tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>")
    tokenizer.fit_on_texts(all_messages)
    
    sequences = tokenizer.texts_to_sequences(all_messages)
    padded = pad_sequences(sequences, maxlen=MAX_LEN, padding='post', truncating='post')
    
    # Split into train/test (80/20)
    split = int(0.8 * len(padded))
    X_train, X_test = padded[:split], padded[split:]
    y_train, y_test = labels[:split], labels[split:]
    
    # Build model
    model = Sequential([
        Embedding(MAX_WORDS, EMBEDDING_DIM, input_length=MAX_LEN),
        GlobalAveragePooling1D(),
        Dense(24, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )
    
    # Train
    model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        validation_data=(X_test, y_test),
        verbose=1
    )
    
    # Evaluate
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nModel accuracy on test set: {accuracy:.4f}")
    
    # Save model
    model.save(MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")
    
    # Save tokenizer config
    with open(TOKENIZER_PATH, 'w', encoding='utf-8') as f:
        json.dump(tokenizer.to_json(), f)
    print(f"Tokenizer saved to {TOKENIZER_PATH}")
    
    return model, tokenizer

if __name__ == "__main__":
    train_and_save_model()