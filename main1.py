import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read the survey data
data = pd.read_csv('survey_data.csv')

# Basic information about the dataset
print(f"Number of survey responses: {len(data)}")
print(f"Age range: {data['Age'].min()} to {data['Age'].max()} years")
print(f"Average age: {data['Age'].mean():.1f} years")

# Low-sodium diet analysis
column_name = [col for col in data.columns if 'low-sodium diet' in col][0]
low_sodium_counts = data[column_name].value_counts()

print("\nLow-sodium diet adoption:")
print(low_sodium_counts)
print(f"Percentage following low-sodium diet: {(low_sodium_counts.get('Yes', 0) / len(data) * 100):.1f}%")

# Visualization for low-sodium diet adoption
plt.figure(figsize=(8, 5))
low_sodium_counts.plot(kind='bar', color=['skyblue', 'salmon'])
plt.title('Low-Sodium Diet Adoption')
plt.xlabel('Diet Adoption')
plt.ylabel('Number of Responses')
plt.xticks(rotation=0)
plt.grid(axis='y')
plt.show()

# Medical conditions for those on low-sodium diet
low_sodium_data = data[data[column_name] == 'Yes']
medical_condition_col = [col for col in data.columns if 'medical condition' in col][0]
medical_condition_counts = low_sodium_data[medical_condition_col].value_counts()
print("\nMedical conditions among low-sodium dieters:")
print(medical_condition_counts)

# Visualization for medical conditions
plt.figure(figsize=(10, 6))
medical_condition_counts.plot(kind='bar', color='lightgreen')
plt.title('Medical Conditions Among Low-Sodium Dieters')
plt.xlabel('Medical Condition')
plt.ylabel('Number of Responses')
plt.xticks(rotation=45)
plt.grid(axis='y')
plt.show()

# Step 1: Strip whitespace from column names
data.columns = data.columns.str.strip()

# Step 2: Print column names for verification
print("Column names in the DataFrame:")
print(data.columns.tolist())

# Step 3: Define a function to safely access columns and print value counts
def safe_value_counts(column_name):
    try:
        return data[column_name].value_counts()
    except KeyError:
        print(f"Column '{column_name}' not found in the DataFrame.")
        return None

# Restaurant dining frequency
restaurant_frequency = safe_value_counts('How often do you eat at restaurants?')
if restaurant_frequency is not None:
    print("\nRestaurant dining frequency:")
    print(restaurant_frequency)

    # Visualization for restaurant dining frequency
    plt.figure(figsize=(8, 5))
    restaurant_frequency.plot(kind='bar', color='coral')
    plt.title('Restaurant Dining Frequency')
    plt.xlabel('Frequency')
    plt.ylabel('Number of Responses')
    plt.xticks(rotation=0)
    plt.grid(axis='y')
    plt.show()

# Satisfaction with low-sodium restaurant options
satisfaction_counts = safe_value_counts('Do you find low-sodium food options at restaurants satisfying in terms of taste?')
if satisfaction_counts is not None:
    print("\nSatisfaction with low-sodium restaurant options:")
    print(satisfaction_counts)

    # Visualization for satisfaction
    plt.figure(figsize=(8, 5))
    satisfaction_counts.plot(kind='bar', color='gold')
    plt.title('Satisfaction with Low-Sodium Restaurant Options')
    plt.xlabel('Satisfaction Level')
    plt.ylabel('Number of Responses')
    plt.xticks(rotation=0)
    plt.grid(axis='y')
    plt.show()

# Interest in taste enhancement device
device_interest = safe_value_counts('Would you be interested in trying a device that enhances salty and umami flavors using electric stimulation?')
if device_interest is not None:
    print("\nInterest in taste enhancement device:")
    print(device_interest)
    print(f"Percentage interested: {(device_interest.get('Yes', 0) / len(data) * 100):.1f}%")

    # Visualization for device interest
    plt.figure(figsize=(8, 5))
    device_interest.plot(kind='bar', color='violet')
    plt.title('Interest in Taste Enhancement Device')
    plt.xlabel('Interest Level')
    plt.ylabel('Number of Responses')
    plt.xticks(rotation=0)
    plt.grid(axis='y')
    plt.show()

# Purchase intention
purchase_intention = safe_value_counts('Would you consider purchasing such a device if it improves taste satisfaction without adding sodium?')
if purchase_intention is not None:
    print("\nPurchase intention:")
    print(purchase_intention)

    # Visualization for purchase intention
    plt.figure(figsize=(8, 5))
    purchase_intention.plot(kind='bar', color='lightblue')
    plt.title('Purchase Intention for Taste Enhancement Device')
    plt.xlabel('Purchase Intention')
    plt.ylabel('Number of Responses')
    plt.xticks(rotation=0)
    plt.grid(axis='y')
    plt.show()

# Importance of taste enhancement
taste_importance = safe_value_counts('How important is taste enhancement in your dining experience?')
if taste_importance is not None:
    print("\nImportance of taste enhancement:")
    print(taste_importance)

    # Visualization for taste importance
    plt.figure(figsize=(8, 5))
    taste_importance.plot(kind='bar', color='peachpuff')
    plt.title('Importance of Taste Enhancement')
    plt.xlabel('Importance Level')
    plt.ylabel('Number of Responses')
    plt.xticks(rotation=0)
    plt.grid(axis='y')
    plt.show()

# Analyze salt content preferences in different dishes
salt_columns = [
    'Dal / Gojju / Palya', 
    'Sambar / Rasam / Curd', 
    'Biryani / Pulao / Rice bath', 
    'Curries (Vegetable/Chicken/Mutton)', 
    'Dosa/Idly/Chaat/Snacks', 
    'Dosa/Idly/Roti/Paratha/Chapathi', 
    'Pickles/Papad'
]

# print("\nSalt content preferences in different dishes:")
# for column in salt_columns:
#     salt_counts = safe_value_counts(column)
#     if salt_counts is not None:
#         print(f"\n{column}:")
#         print(salt_counts)

#         # Visualization for salt content preferences
#         plt.figure(figsize=(10, 6))
#         salt_counts.plot(kind='bar', color='lightcoral')
#         plt.title(f'Salt Content Preferences in {column}')
#         plt.xlabel('Preference Level')
#         plt.ylabel('Number of Responses')
#         plt.xticks(rotation=0)
#         plt.grid(axis='y')
#         plt.show()

# Overall perception of current salt content
overall_perception = safe_value_counts('Do you find the current salt content in these dishes')
if overall_perception is not None:
    print("\nOverall perception of current salt content:")
    print(overall_perception)

    # Visualization for overall perception
    plt.figure(figsize=(8, 5))
    overall_perception.plot(kind='bar', color='lightgreen')
    plt.title('Overall Perception of Current Salt Content')
    plt.xlabel('Perception Level')
    plt.ylabel('Number of Responses')
    plt.xticks(rotation=0)
    plt.grid(axis='y')
    plt.show()
def load_user_health_profiles():
    try:
        # Update this path to where you've saved the CSV file
        file_path = 'user_profile.csv'  # Replace with your actual path
        health_df = pd.read_csv(file_path)
        health_profiles_dict = {}
        
        for _, row in health_df.iterrows():
            health_profiles_dict[row['user_id']] = {
                'condition': row['condition'],
                'severity': row['severity'],
                'sodium_restriction_mg': row['sodium_restriction_mg'],
                'taste_sensitivity': row['taste_sensitivity'],
                'age': row['age'],
                'gender': row['gender']
            }
        
        return health_profiles_dict
    except FileNotFoundError:
        print("User health profiles file not found. Creating default profile.")
        return {1: {'condition': 'none', 'severity': 'none', 'sodium_restriction_mg': 2300, 
                   'taste_sensitivity': 'normal', 'age': 35, 'gender': 'M'}}
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt

# Define base path - update this to match your dataset structure
BASE_DIR = r'C:\Users\patel\Downloads\archive(1)\Indian_Food_Images'

# Set parameters
IMG_SIZE = 224  # MobileNetV2 default input size
BATCH_SIZE = 32
EPOCHS = 10
NUM_CLASSES = 20  # Update this based on your dataset

# Load your salt content dataset
def load_salt_data():
    salt_df = pd.read_csv('salt_data.csv')
    salt_content_dict = {}
    
    for _, row in salt_df.iterrows():
        food_key = row['Food_Name'].lower().replace(' ', '_')
        salt_content_dict[food_key] = {
            'salt_content_mg': row['Salt_Content_mg_per_100g'],
            'category': row['Category'],
            'region': row['Region'],
            'main_ingredients': row['Main_Ingredients'],
            'typical_portion_size_g': row['Typical_Portion_Size_g'],
            'salt_per_typical_portion_mg': row['Salt_Per_Typical_Portion_mg']
        }
    
    return salt_content_dict

# Load the salt content data
indian_food_salt_content = load_salt_data()

# Function to create and compile the CNN model
def create_food_recognition_model():
    base_model = MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights='imagenet'
    )
    
    base_model.trainable = False
    
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(NUM_CLASSES, activation='softmax')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# Data augmentation for training
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Only rescale for validation/testing
val_datagen = ImageDataGenerator(rescale=1./255)

# Load and preprocess the data
def load_data():
    # Create a list of all image directories
    food_classes = os.listdir(BASE_DIR)
    food_classes = [f for f in food_classes if os.path.isdir(os.path.join(BASE_DIR, f))]
    
    # Split data into training and validation sets
    train_data = []
    val_data = []
    
    for food_class in food_classes:
        food_path = os.path.join(BASE_DIR, food_class)
        images = os.listdir(food_path)
        np.random.shuffle(images)
        
        split_index = int(len(images) * 0.8)  # 80% for training, 20% for validation
        train_data.extend([(os.path.join(food_path, img), food_class) for img in images[:split_index]])
        val_data.extend([(os.path.join(food_path, img), food_class) for img in images[split_index:]])
    
    # Create generators
    train_generator = train_datagen.flow_from_dataframe(
        pd.DataFrame(train_data, columns=['filename', 'class']),
        x_col='filename',
        y_col='class',
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical'
    )

    validation_generator = val_datagen.flow_from_dataframe(
        pd.DataFrame(val_data, columns=['filename', 'class']),
        x_col='filename',
        y_col='class',
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical'
    )
    
    return train_generator, validation_generator

# Create and train the model
def train_model(train_generator, validation_generator):
    model = create_food_recognition_model()
    
    checkpoint = ModelCheckpoint(
        'best_food_model.h5',
        monitor='val_accuracy',
        save_best_only=True,
        mode='max'
    )
    
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )
    
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // BATCH_SIZE,
        callbacks=[checkpoint, early_stopping]
    )
    
    return model, history

# Plot training history
def plot_training_history(history):
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()

    
# Enhanced personalized stimulation adjustment based on user health profile and food salt content
def personalized_stimulation(salt_content, user_id=1):
    """
    Adjusts the stimulation level based on the salt content and user's health profile.
    
    Parameters:
    salt_content (float): The salt content in mg per 100g.
    user_id (int): The ID of the user for personalized stimulation.
    
    Returns:
    float: The adjusted stimulation level on a scale of 0 to 10.
    """
    # Get user health profile
    if user_id in user_health_profiles:
        user_profile = user_health_profiles[user_id]
    else:
        # Default profile if user ID not found
        user_profile = {
            'condition': 'none', 
            'severity': 'none',
            'sodium_restriction_mg': 2300,  # Default sodium restriction for healthy adults
            'taste_sensitivity': 'normal'
        }
    
    # Different logic for users with no health conditions
    if user_profile['condition'] == 'none':
        # For healthy users, stimulation is based more on taste preference than medical need
        
        # Base stimulation based on salt content for healthy users
        # Lower stimulation for high-salt foods (letting natural taste come through)
        # Higher stimulation for low-salt foods (enhancing subtle flavors)
        if salt_content > 500:  # High salt content
            base_stimulation = 2.0
        elif salt_content > 300:  # Medium salt content
            base_stimulation = 3.0
        elif salt_content > 150:  # Low salt content
            base_stimulation = 4.0
        else:  # Very low salt content
            base_stimulation = 5.0
            
        # Taste sensitivity adjustments for healthy users
        sensitivity_adjustment = {
            'highly_reduced': 2.0,      # Needs much more stimulation
            'reduced': 1.5,             # Needs more stimulation
            'slightly_reduced': 1.0,    # Needs slightly more stimulation
            'normal': 0,                # No adjustment
            'slightly_enhanced': -1.0,  # Needs less stimulation
            'enhanced': -1.5            # Needs much less stimulation
        }
        
        # Age-based adjustment (older people often have reduced taste sensitivity)
        age = user_profile['age']
        if age > 60:
            age_adjustment = 1.0
        elif age > 50:
            age_adjustment = 0.5
        elif age > 40:
            age_adjustment = 0.2
        else:
            age_adjustment = 0
            
        # Calculate final stimulation for healthy users
        final_stimulation = (
            base_stimulation + 
            sensitivity_adjustment.get(user_profile['taste_sensitivity'], 0) +
            age_adjustment
        )
        
    else:
        # Original logic for users with health conditions
        
        # Calculate base stimulation based on the ratio of salt content to user's sodium restriction
        daily_limit = user_profile['sodium_restriction_mg']
        portion_salt_percentage = (salt_content / daily_limit) * 100
        
        # Base stimulation calculation
        # Higher salt content relative to restriction = higher stimulation needed
        if portion_salt_percentage > 30:  # Very high salt content relative to restriction
            base_stimulation = 8.0
        elif portion_salt_percentage > 20:  # High salt content
            base_stimulation = 6.5
        elif portion_salt_percentage > 10:  # Moderate salt content
            base_stimulation = 5.0
        elif portion_salt_percentage > 5:   # Low salt content
            base_stimulation = 3.5
        else:  # Very low salt content
            base_stimulation = 2.0
        
        # Adjust for condition severity
        severity_adjustment = {
            'none': 0,
            'mild': 0.5,
            'moderate': 1.0,
            'severe': 1.5
        }
        
        # Adjust for taste sensitivity
        sensitivity_adjustment = {
            'highly_reduced': 1.0,     # Needs more stimulation
            'reduced': 0.5,            # Needs slightly more stimulation
            'normal': 0,               # No adjustment
            'enhanced': -0.5           # Needs less stimulation
        }
        
        # Condition-specific adjustments
        condition_adjustment = 0
        if user_profile['condition'] == 'hypertension':
            # Hypertension patients need more taste enhancement for low-salt foods
            condition_adjustment = 0.5
        elif user_profile['condition'] == 'kidney_disease':
            # Kidney disease patients often have altered taste perception
            condition_adjustment = 0.7
        elif user_profile['condition'] == 'heart_failure':
            # Heart failure patients typically need strict sodium restrictions
            condition_adjustment = 0.6
        elif user_profile['condition'] == 'liver_disease':
            # Liver disease can affect taste perception
            condition_adjustment = 0.4
        
        # Calculate final stimulation level with all adjustments
        final_stimulation = (
            base_stimulation + 
            severity_adjustment.get(user_profile['severity'], 0) +
            sensitivity_adjustment.get(user_profile['taste_sensitivity'], 0) +
            condition_adjustment
        )
    
    # Ensure the stimulation level is within the valid range (0-10)
    final_stimulation = max(0, min(10, final_stimulation))
    
    # Round to 2 decimal places for precision
    return round(final_stimulation, 2)


# Generate personalized dietary and usage recommendations
def generate_recommendations(food_name, nutrition_data, user_id=1):
    """
    Generates personalized dietary recommendations based on the food's nutritional information
    and the user's health profile.
    
    Parameters:
    food_name (str): The name of the food.
    nutrition_data (dict): A dictionary containing the food's nutritional information.
    user_id (int): The ID of the user for personalized recommendations.
    
    Returns:
    list: A list of recommendation strings.
    """
    # Get user health profile
    if user_id in user_health_profiles:
        user_profile = user_health_profiles[user_id]
    else:
        # Default profile if user ID not found
        user_profile = {
            'condition': 'none', 
            'severity': 'none',
            'sodium_restriction_mg': 2300,  # Default sodium restriction
            'taste_sensitivity': 'normal'
        }
    
    recommendations = []
    
    # Basic food information
    recommendations.append(f"{food_name} contains approximately {nutrition_data['salt_content_mg']} mg of salt per 100g.")
    
    # Typical portion information
    recommendations.append(f"A typical portion ({nutrition_data['typical_portion_size_g']}g) contains {nutrition_data['salt_per_typical_portion_mg']} mg of salt.")
    
    # Different recommendations for users with and without health conditions
    if user_profile['condition'] == 'none':
        # For users without health conditions
        daily_salt_limit = 2300  # General recommendation for healthy adults
        salt_percent_daily = (nutrition_data['salt_per_typical_portion_mg'] / daily_salt_limit) * 100
        
        recommendations.append(f"This represents approximately {salt_percent_daily:.1f}% of the general recommended daily salt intake (2300 mg).")
        
        # General healthy eating advice
        if salt_percent_daily > 30:
            recommendations.append("This dish has high salt content. Consider balancing your salt intake for the rest of the day.")
        
        # Smart Spoon recommendations based on taste preference
        if user_profile['taste_sensitivity'] == 'enhanced':
            recommendations.append("Based on your enhanced taste sensitivity, a lower stimulation level is recommended.")
        elif user_profile['taste_sensitivity'] in ['reduced', 'highly_reduced']:
            recommendations.append("Based on your reduced taste sensitivity, a higher stimulation level is recommended to enhance flavor perception.")
        
        # Add age-specific recommendations if older
        if user_profile['age'] > 60:
            recommendations.append("As taste sensitivity may naturally decrease with age, you might benefit from higher stimulation levels.")
        
    else:
        # For users with health conditions (original code)
        # Personalized daily intake percentage
        daily_salt_limit = user_profile['sodium_restriction_mg']
        salt_percent_daily = (nutrition_data['salt_per_typical_portion_mg'] / daily_salt_limit) * 100
        recommendations.append(f"This represents approximately {salt_percent_daily:.1f}% of your recommended daily salt limit ({daily_salt_limit} mg).")
        
        # Condition-specific recommendations
        if user_profile['condition'] == 'hypertension':
            if user_profile['severity'] in ['moderate', 'severe']:
                if salt_percent_daily > 15:
                    recommendations.append("CAUTION: This dish exceeds the recommended salt intake for your hypertension condition.")
                    recommendations.append("Consider reducing portion size or choosing a lower-sodium alternative.")
                else:
                    recommendations.append("This dish is within acceptable sodium limits for your condition.")
        
        elif user_profile['condition'] == 'kidney_disease':
            if salt_percent_daily > 10:
                recommendations.append("CAUTION: This dish may not be suitable for your kidney condition.")
                recommendations.append("Consider speaking with your dietitian before consuming this dish.")
            else:
                recommendations.append("This dish is within acceptable sodium limits for your condition.")
        
        elif user_profile['condition'] == 'heart_failure':
            if salt_percent_daily > 12:
                recommendations.append("CAUTION: This dish exceeds the recommended salt intake for heart failure patients.")
                recommendations.append("Consider reducing portion size or choosing a lower-sodium alternative.")
            else:
                recommendations.append("This dish is within acceptable sodium limits for your condition.")
        
        # Taste sensitivity recommendations
        if user_profile['taste_sensitivity'] == 'reduced' or user_profile['taste_sensitivity'] == 'highly_reduced':
            recommendations.append("Your taste sensitivity profile suggests you may benefit from enhanced flavor stimulation.")
    
    # General recommendations for everyone
    if nutrition_data['salt_content_mg'] > 500:
        recommendations.append("Consider pairing this dish with fresh vegetables or a simple salad to balance salt intake.")
    
    # Smart Spoon usage recommendation
    recommendations.append(f"Recommended Smart Spoon stimulation level: {personalized_stimulation(nutrition_data['salt_content_mg'], user_id):.2f}/10")
    
    return recommendations


import matplotlib.pyplot as plt

def visualize_salt_content(salt_content, food_name):
    """
    Visualizes the salt content of a food item.
    
    Parameters:
    salt_content (float): The salt content in mg per 100g.
    food_name (str): The name of the food item.
    """
    # Define the salt content thresholds for different visualization levels
    low_threshold = 120
    medium_threshold = 480
    high_threshold = 800
    
    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot the salt content bar
    ax.bar(0, salt_content, color='blue')
    
    # Set the x-axis labels and title
    ax.set_xticks([0])
    ax.set_xticklabels([food_name])
    ax.set_title(f"Salt Content of {food_name}")
    
    # Set the y-axis label
    ax.set_ylabel("Salt Content (mg per 100g)")
    
    # Add horizontal lines for the salt content thresholds
    ax.axhline(y=low_threshold, color='green', linestyle='--', label='Low')
    ax.axhline(y=medium_threshold, color='yellow', linestyle='--', label='Medium')
    ax.axhline(y=high_threshold, color='red', linestyle='--', label='High')
    
    # Add a legend
    ax.legend()
    
    # Display the plot
    plt.show()
    
    
# (The rest of your functions remain unchanged)
def get_food_nutrition_data(predicted_class, class_indices):
    # Reverse the class indices to get food name from the class index
    food_name = list(class_indices.keys())[predicted_class]
    
    # Convert food name to the format used in salt content dictionary
    food_key = food_name.lower().replace(' ', '_')
    
    # Retrieve the nutrition data from the salt content dictionary
    if food_key in indian_food_salt_content:
        nutrition_data = indian_food_salt_content[food_key]
        return nutrition_data, food_name
    else:
        raise ValueError(f"Nutrition data for {food_name} not found.")

# Main prediction function
def predict_and_process_image(model, image_path, class_indices,user_id):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class]
    
    nutrition_data, food_name = get_food_nutrition_data(predicted_class, class_indices)
    salt_content = nutrition_data['salt_content_mg']
    stimulation = personalized_stimulation(salt_content, user_id)
    recommendations = generate_recommendations(food_name, nutrition_data,user_id)
    
    result = {
        'food_name': food_name.replace('_', ' ').title(),
        'confidence': float(confidence),
        'salt_content_mg': salt_content,
        'salt_per_portion_mg': nutrition_data['salt_per_typical_portion_mg'],
        'typical_portion_size_g': nutrition_data['typical_portion_size_g'],
        'category': nutrition_data['category'],
        'region': nutrition_data['region'],
        'main_ingredients': nutrition_data['main_ingredients'],
        'stimulation_level': stimulation,
        'recommendations': recommendations
    }
    
    return result
def main():
    # Load the necessary data
    train_generator, validation_generator = load_data()
    class_indices = train_generator.class_indices
    
    # Load user health profiles data - add this line
    global user_health_profiles
    user_health_profiles = load_user_health_profiles()
    
    # Load or train model
    try:
        model = load_model('best_food_model.h5')
        print("Loaded existing model")
    except:
        print("Training new model...")
        model, history = train_model(train_generator, validation_generator)
        plot_training_history(history)
    
    # Example prediction with a test image
    test_image_path = "test6_img.jpg" # Update this with a valid test image path
    
    # Ask for user ID - add this section
    try:
        user_id = int(input("Enter user ID (1-25) or press Enter for default user ID 1: ") or 1)
        if user_id not in user_health_profiles:
            print(f"User ID {user_id} not found. Using default user ID 1.")
            user_id = 1
    except ValueError:
        print("Invalid input. Using default user ID 1.")
        user_id = 1
    
    # Pass the user_id to predict_and_process_image
    result = predict_and_process_image(model, test_image_path, class_indices, user_id)
    
    # Print results
    print("\n===== FOOD ANALYSIS RESULTS =====")
    print(f"Recognized Food: {result['food_name']} (Confidence: {result['confidence']:.2f})")
    print(f"Category: {result['category']} | Region: {result['region']}")
    print(f"Main Ingredients: {result['main_ingredients']}")
    print(f"Salt Content: {result['salt_content_mg']} mg per 100g")
    print(f"Typical Portion: {result['typical_portion_size_g']}g containing {result['salt_per_portion_mg']} mg salt")
    
    # Add user condition information - add these lines
    if result.get('user_condition', "None") != "None":
        print(f"User Health Condition: {result['user_condition']} ({result['user_condition_severity']})")
    
    print(f"Recommended Stimulation Level: {result['stimulation_level']:.2f}/10")
    
    visualize_salt_content(result['salt_content_mg'], result['food_name'])
    
    print("\nDietary Recommendations:")
    for i, rec in enumerate(result['recommendations'], 1):
        print(f"{i}. {rec}")

if __name__ == "__main__":
    main()
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

# Download NLTK resources
nltk.download('vader_lexicon')

# Load the dataset with a specific encoding
data = pd.read_csv('response_data.csv', encoding='ISO-8859-1')  # Try 'utf-16' or 'cp1252' if needed

# Step 1: Sentiment Analysis using NLTK
sia = SentimentIntensityAnalyzer()

# Create a new column for sentiment scores
data['Sentiment_Score'] = data['Feedback_Text'].apply(lambda x: sia.polarity_scores(x)['compound'])

# Classify sentiment based on the score
data['Sentiment'] = data['Sentiment_Score'].apply(lambda x: 'Positive' if x > 0.05 else ('Negative' if x < -0.05 else 'Neutral'))

# Step 2: Prepare for Predicting User Behavior
# Convert satisfaction levels to categorical for modeling
data['Satisfaction_Level'] = data['Satisfaction_Level'].astype(str)

# Text Vectorization
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(data['Feedback_Text'])
y = data['Satisfaction_Level']

# Step 3: Train a Naive Bayes classifier
model = MultinomialNB()
model.fit(X, y)

# Function to analyze user input and generate improvement suggestions
def analyze_feedback(user_feedback):
    # Sentiment Analysis
    sentiment_score = sia.polarity_scores(user_feedback)['compound']
    sentiment = 'Positive' if sentiment_score > 0.05 else ('Negative' if sentiment_score < -0.05 else 'Neutral')
    
    # Vectorize user feedback for prediction
    user_feedback_vec = vectorizer.transform([user_feedback])
    
    # Predict satisfaction level
    predicted_satisfaction = model.predict(user_feedback_vec)[0]
    
    # Generate improvement suggestions based on sentiment and keywords
    improvement_suggestions = []
    
    # Define a mapping of keywords to specific suggestions
    suggestions_map = {
        'battery': [
            "Consider increasing the battery capacity for longer usage.",
            "Implement a low-power mode to extend battery life."
        ],
        'grip': [
            "Enhance the ergonomic design for a better grip.",
            "Use non-slip materials to improve handling."
        ],
        'customizable': [
            "Add more customizable settings tailored to user preferences.",
            "Introduce presets for different types of food."
        ],
        'quality': [
            "Maintain the high-quality standards in materials and design.",
            "Conduct regular quality checks to ensure product reliability."
        ],
        'features': [
            "Explore user-requested features like temperature control or smart alerts.",
            "Consider integrating a companion app for enhanced functionality."
        ],
        'color': [
            "Introduce a variety of color options for personalization.",
            "Offer limited edition colors based on user feedback."
        ],
        'nutrition': [
            "Implement tracking features for nutritional intake.",
            "Consider adding a recipe suggestion feature based on dietary needs."
        ],
        'price': [
            "Evaluate pricing strategies to ensure competitiveness.",
            "Consider offering bundle deals for multiple purchases."
        ],
        'salt': [
            "Adjust the salt levels based on user feedback to improve taste.",
            "Consider offering options for low-sodium alternatives.",
            "Implement a feature for users to customize their preferred salt level."
        ]
    }
    
    # Check for keywords in user feedback and append relevant suggestions
    for keyword, suggestions in suggestions_map.items():
        if keyword in user_feedback.lower():
            improvement_suggestions.extend(suggestions)
    
    # Fallback suggestion if no specific keywords were found
    if not improvement_suggestions:
        improvement_suggestions.append("Thank you for your feedback! We will consider your suggestions for future improvements.")

    return {
        'Sentiment': sentiment,
        'Predicted Satisfaction Level': predicted_satisfaction,
        'Improvement Suggestions': improvement_suggestions
    }

# User input
user_review = input("Please enter your feedback about the Smart Spoon: ")
result = analyze_feedback(user_review)

# Display the results
print("\nAnalysis Results:")
print(f"Sentiment: {result['Sentiment']}")
print(f"Predicted Satisfaction Level: {result['Predicted Satisfaction Level']}")
print("Improvement Suggestions:")
for suggestion in result['Improvement Suggestions']:
    print(f"- {suggestion}")
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.cluster import KMeans
from gensim.models import Word2Vec
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from transformers import pipeline
import warnings
warnings.filterwarnings('ignore')

# Download necessary NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# Load the dataset with appropriate encoding
try:
    data = pd.read_csv('response_data.csv', encoding='utf-8')
except UnicodeDecodeError:
    try:
        data = pd.read_csv('response_data.csv', encoding='ISO-8859-1')
    except UnicodeDecodeError:
        data = pd.read_csv('response_data.csv', encoding='cp1252')

print(f"Loaded dataset with {len(data)} records")

# Data preprocessing function
def preprocess_text(text):
    if isinstance(text, str):
        # Convert to lowercase
        text = text.lower()
        # Remove special characters and numbers
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\d+', '', text)
        # Tokenize
        tokens = nltk.word_tokenize(text)
        # Remove stopwords
        stop_words = set(stopwords.words('english'))
        tokens = [word for word in tokens if word not in stop_words]
        # Lemmatization
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(word) for word in tokens]
        return ' '.join(tokens)
    return ""

# Apply preprocessing to feedback text
print("Preprocessing text data...")
data['Processed_Text'] = data['Feedback_Text'].apply(preprocess_text)

# Create a list of tokenized words for Word2Vec
tokenized_texts = [text.split() for text in data['Processed_Text'] if isinstance(text, str) and text.strip()]

# Word2Vec for word embeddings - capture semantic relationships
print("Training Word2Vec model...")
word2vec_model = Word2Vec(
    tokenized_texts,
    vector_size=100,
    window=5,
    min_count=2,
    workers=4
)

# Function to get document vectors from Word2Vec
def document_vector(word2vec_model, doc):
    # Remove out-of-vocabulary words
    doc = [word for word in doc if word in word2vec_model.wv.key_to_index]
    if len(doc) == 0:
        return np.zeros(word2vec_model.vector_size)
    return np.mean(word2vec_model.wv[doc], axis=0)

# Create document vectors
doc_vectors = []
for tokens in tokenized_texts:
    doc_vectors.append(document_vector(word2vec_model, tokens))

# Create a DataFrame with the vectors for clustering
print("Performing user feedback clustering...")
doc_vectors_df = pd.DataFrame(doc_vectors)

# K-means clustering to identify patterns in feedback
kmeans = KMeans(n_clusters=5, random_state=42)
if len(doc_vectors) > 0:
    data_subset = data.iloc[:len(doc_vectors)].copy()
    data_subset['Cluster'] = kmeans.fit_predict(doc_vectors_df)
    
    # Analyze clusters
    cluster_analysis = data_subset.groupby('Cluster').agg({
        'Satisfaction_Level': 'mean',
        'Feedback_Text': lambda x: ' '.join(x)
    }).reset_index()
    
    # Get top keywords for each cluster
    cluster_analysis['Top_Keywords'] = cluster_analysis['Feedback_Text'].apply(
        lambda x: ' '.join([word for word, count in nltk.FreqDist(nltk.word_tokenize(preprocess_text(x))).most_common(5)])
    )
    
    print("\nCluster Analysis:")
    print(cluster_analysis[['Cluster', 'Satisfaction_Level', 'Top_Keywords']])

# Advanced sentiment analysis using HuggingFace Transformers
print("Performing advanced sentiment analysis...")
sentiment_analyzer = pipeline('sentiment-analysis')

# Apply sentiment analysis to a sample of the data (to avoid rate limits or performance issues)
sample_size = min(100, len(data))
sample_indices = np.random.choice(len(data), sample_size, replace=False)
data['Advanced_Sentiment'] = None

for idx in sample_indices:
    text = data.loc[idx, 'Feedback_Text']
    if isinstance(text, str) and text.strip():
        try:
            result = sentiment_analyzer(text)
            data.loc[idx, 'Advanced_Sentiment'] = result[0]['label']
        except Exception as e:
            print(f"Error analyzing sentiment for index {idx}: {e}")

# TF-IDF Vectorization for better feature extraction
print("Creating TF-IDF features...")
tfidf_vectorizer = TfidfVectorizer(
    max_features=1000,
    min_df=2,
    max_df=0.85
)
tfidf_features = tfidf_vectorizer.fit_transform(data['Processed_Text'].fillna(''))

# Prepare data for satisfaction level prediction
X = tfidf_features
y = data['Satisfaction_Level'].astype(str)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a more advanced RandomForest model for satisfaction prediction
print("Training RandomForest classifier...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Evaluate the model
y_pred = rf_model.predict(X_test)
print("\nModel Evaluation:")
print(classification_report(y_test, y_pred))

# Feature importance analysis
feature_importance = pd.DataFrame({
    'Feature': tfidf_vectorizer.get_feature_names_out(),
    'Importance': rf_model.feature_importances_
}).sort_values(by='Importance', ascending=False)

print("\nTop features affecting satisfaction:")
print(feature_importance.head(10))

# Extract key improvement areas based on important features and negative sentiment
def extract_improvement_areas(data, feature_importance, top_n=20):
    # Identify samples with lower satisfaction levels
    low_satisfaction = data[data['Satisfaction_Level'].astype(float) < 3.5]
    
    # Get important words from feature importance
    important_words = set(feature_importance.head(top_n)['Feature'])
    
    # Look for these words in low satisfaction feedback
    improvement_areas = {}
    
    for _, row in low_satisfaction.iterrows():
        feedback = row['Processed_Text']
        if isinstance(feedback, str):
            for word in important_words:
                if word in feedback:
                    if word in improvement_areas:
                        improvement_areas[word] += 1
                    else:
                        improvement_areas[word] = 1
    
    return sorted(improvement_areas.items(), key=lambda x: x[1], reverse=True)

improvement_areas = extract_improvement_areas(data, feature_importance)
print("\nKey improvement areas based on analysis:")
for area, count in improvement_areas[:10]:
    print(f"- {area}: {count} mentions in negative feedback")

# Create a suggestions database based on analysis
suggestions_map = {
    'battery': [
        "Increase battery capacity based on user feedback analysis.",
        "Implement smart power management optimized for different usage patterns.",
        "Add battery status indicators with predictive time remaining."
    ],
    'grip': [
        "Redesign ergonomic grip based on hand size clustering from user data.",
        "Implement adjustable grip options for different user preferences."
    ],
    'weight': [
        "Optimize weight distribution based on usage pattern analysis.",
        "Offer lightweight alternatives identified through sentiment patterns."
    ],
    'app': [
        "Enhance app connectivity based on user interaction data.",
        "Add features users frequently request in feedback (smart alerts, nutrition tracking)."
    ],
    'temperature': [
        "Implement adaptive temperature control based on usage patterns.",
        "Add temperature presets derived from positive feedback clusters."
    ],
    'sync': [
        "Improve synchronization reliability based on pain points identified.",
        "Develop auto-reconnect features based on usage pattern analysis."
    ],
    'customizable': [
        "Expand customization options based on feature importance analysis.",
        "Create user-specific presets derived from satisfaction predictors."
    ],
    'quality': [
        "Focus on quality improvements in areas most correlated with satisfaction.",
        "Implement quality control checks for features mentioned in negative clusters."
    ],
    'price': [
        "Analyze price-to-satisfaction ratio for different user segments.",
        "Develop targeted pricing strategies based on satisfaction predictors."
    ],
    'salt': [
        "Implement adaptive salt dispensing based on taste preference clusters.",
        "Develop salt level customization based on sentiment analysis findings."
    ]
}

# Add dynamic generation of suggestions based on analysis
for word, _ in improvement_areas[:5]:
    if word not in suggestions_map and len(word) > 3:  # Filter out short words
        related_words = []
        if word in word2vec_model.wv:
            related_words = [w for w, _ in word2vec_model.wv.most_similar(word, topn=3)]
        
        suggestions_map[word] = [
            f"Improve {word} functionality based on user feedback analysis.",
            f"Consider redesigning {word} features mentioned in negative reviews.",
            f"Research alternative approaches to {word} related features."
        ]

# Main analysis function for user feedback
def analyze_feedback(user_feedback, word2vec_model, tfidf_vectorizer, rf_model, sentiment_analyzer, suggestions_map):
    # Preprocess the feedback
    processed_feedback = preprocess_text(user_feedback)
    
    # Advanced sentiment analysis
    try:
        sentiment_result = sentiment_analyzer(user_feedback)
        sentiment = sentiment_result[0]['label']
        sentiment_score = sentiment_result[0]['score']
    except:
        # Fallback to a simpler approach if transformer fails
        sentiment = "NEUTRAL"
        sentiment_score = 0.5
    
    # Vectorize for prediction
    user_feedback_vec = tfidf_vectorizer.transform([processed_feedback])
    
    # Predict satisfaction level
    predicted_satisfaction = rf_model.predict(user_feedback_vec)[0]
    
    # Get document embedding for similarity analysis
    tokens = processed_feedback.split()
    doc_vec = document_vector(word2vec_model, tokens)
    
    # Generate improvement suggestions based on feedback content
    improvement_suggestions = []
    
    # Check for keywords in processed feedback
    for keyword, suggestions in suggestions_map.items():
        if keyword in processed_feedback:
            # Choose the most relevant suggestion based on word similarity
            if len(suggestions) > 1 and keyword in word2vec_model.wv:
                # Find most similar words to the feedback
                feedback_tokens = set(processed_feedback.split())
                most_similar_idx = 0
                highest_similarity = -1
                
                for i, suggestion in enumerate(suggestions):
                    # Calculate similarity between suggestion and feedback
                    suggestion_tokens = set(preprocess_text(suggestion).split())
                    overlap = feedback_tokens.intersection(suggestion_tokens)
                    similarity = len(overlap) / max(1, len(feedback_tokens.union(suggestion_tokens)))
                    
                    if similarity > highest_similarity:
                        highest_similarity = similarity
                        most_similar_idx = i
                
                improvement_suggestions.append(suggestions[most_similar_idx])
            else:
                improvement_suggestions.append(suggestions[0])
    
    # If no specific keywords were found, generate suggestions based on sentiment
    if not improvement_suggestions:
        if sentiment == "NEGATIVE":
            # Extract important words from the feedback
            feedback_words = set(processed_feedback.split())
            for word in feedback_words:
                if len(word) > 3 and word in feature_importance['Feature'].values:
                    improvement_suggestions.append(f"Consider improving the {word} feature based on user feedback analysis.")
                    break
        
        if not improvement_suggestions:
            improvement_suggestions.append("Thank you for your feedback! Our AI has recorded your comments for future product improvements.")

    # Limit to top 3 most relevant suggestions
    improvement_suggestions = improvement_suggestions[:3]

    return {
        'Sentiment': sentiment,
        'Sentiment_Score': sentiment_score,
        'Predicted_Satisfaction_Level': predicted_satisfaction,
        'Improvement_Suggestions': improvement_suggestions
    }

# Example usage
if __name__ == "__main__":
    print("\n" + "="*50)
    print("Smart Spoon Feedback Analysis System")
    print("="*50)
    
    user_review = input("\nPlease enter your feedback about the Smart Spoon: ")
    
    result = analyze_feedback(
        user_review, 
        word2vec_model, 
        tfidf_vectorizer, 
        rf_model,
        sentiment_analyzer,
        suggestions_map
    )

    # Display the results
    print("\nAnalysis Results:")
    print(f"Sentiment: {result['Sentiment']} (confidence: {result['Sentiment_Score']:.2f})")
    print(f"Predicted Satisfaction Level: {result['Predicted_Satisfaction_Level']}")
    print("\nPersonalized Improvement Suggestions:")
    for suggestion in result['Improvement_Suggestions']:
        print(f"- {suggestion}")
    
    print("\nThank you for helping us improve the Smart Spoon!")
pip install wordcloud
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.cluster import KMeans
from gensim.models import Word2Vec
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from transformers import pipeline
import warnings
warnings.filterwarnings('ignore')

# Download necessary NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# Load the dataset with appropriate encoding
try:
    data = pd.read_csv('response_data.csv', encoding='utf-8')
except UnicodeDecodeError:
    try:
        data = pd.read_csv('response_data.csv', encoding='ISO-8859-1')
    except UnicodeDecodeError:
        data = pd.read_csv('response_data.csv', encoding='cp1252')

print(f"Loaded dataset with {len(data)} records")

# Data preprocessing function
def preprocess_text(text):
    if isinstance(text, str):
        # Convert to lowercase
        text = text.lower()
        # Remove special characters and numbers
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\d+', '', text)
        # Tokenize
        tokens = nltk.word_tokenize(text)
        # Remove stopwords
        stop_words = set(stopwords.words('english'))
        tokens = [word for word in tokens if word not in stop_words]
        # Lemmatization
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(word) for word in tokens]
        return ' '.join(tokens)
    return ""

# Apply preprocessing to feedback text
print("Preprocessing text data...")
data['Processed_Text'] = data['Feedback_Text'].apply(preprocess_text)

# Create a list of tokenized words for Word2Vec
tokenized_texts = [text.split() for text in data['Processed_Text'] if isinstance(text, str) and text.strip()]

# Word2Vec for word embeddings - capture semantic relationships
print("Training Word2Vec model...")
word2vec_model = Word2Vec(
    tokenized_texts,
    vector_size=100,
    window=5,
    min_count=2,
    workers=4
)

# Function to get document vectors from Word2Vec
def document_vector(word2vec_model, doc):
    # Remove out-of-vocabulary words
    doc = [word for word in doc if word in word2vec_model.wv.key_to_index]
    if len(doc) == 0:
        return np.zeros(word2vec_model.vector_size)
    return np.mean(word2vec_model.wv[doc], axis=0)

# Create document vectors
doc_vectors = []
for tokens in tokenized_texts:
    doc_vectors.append(document_vector(word2vec_model, tokens))

# Create a DataFrame with the vectors for clustering
print("Performing user feedback clustering...")
doc_vectors_df = pd.DataFrame(doc_vectors)

# K-means clustering to identify patterns in feedback
kmeans = KMeans(n_clusters=5, random_state=42)
if len(doc_vectors) > 0:
    data_subset = data.iloc[:len(doc_vectors)].copy()
    data_subset['Cluster'] = kmeans.fit_predict(doc_vectors_df)
    
    # Analyze clusters
    cluster_analysis = data_subset.groupby('Cluster').agg({
        'Satisfaction_Level': 'mean',
        'Feedback_Text': lambda x: ' '.join(x)
    }).reset_index()
    
    # Get top keywords for each cluster
    cluster_analysis['Top_Keywords'] = cluster_analysis['Feedback_Text'].apply(
        lambda x: ' '.join([word for word, count in nltk.FreqDist(nltk.word_tokenize(preprocess_text(x))).most_common(5)])
    )
    
    print("\nCluster Analysis:")
    print(cluster_analysis[['Cluster', 'Satisfaction_Level', 'Top_Keywords']])

# Advanced sentiment analysis using HuggingFace Transformers
print("Performing advanced sentiment analysis...")
sentiment_analyzer = pipeline('sentiment-analysis')

# Apply sentiment analysis to a sample of the data (to avoid rate limits or performance issues)
sample_size = min(100, len(data))
sample_indices = np.random.choice(len(data), sample_size, replace=False)
data['Advanced_Sentiment'] = None

for idx in sample_indices:
    text = data.loc[idx, 'Feedback_Text']
    if isinstance(text, str) and text.strip():
        try:
            result = sentiment_analyzer(text)
            data.loc[idx, 'Advanced_Sentiment'] = result[0]['label']
        except Exception as e:
            print(f"Error analyzing sentiment for index {idx}: {e}")

# TF-IDF Vectorization for better feature extraction
print("Creating TF-IDF features...")
tfidf_vectorizer = TfidfVectorizer(
    max_features=1000,
    min_df=2,
    max_df=0.85
)
tfidf_features = tfidf_vectorizer.fit_transform(data['Processed_Text'].fillna(''))

# Prepare data for satisfaction level prediction
X = tfidf_features
y = data['Satisfaction_Level'].astype(str)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a more advanced RandomForest model for satisfaction prediction
print("Training RandomForest classifier...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Evaluate the model
y_pred = rf_model.predict(X_test)
print("\nModel Evaluation:")
print(classification_report(y_test, y_pred))

# Feature importance analysis
feature_importance = pd.DataFrame({
    'Feature': tfidf_vectorizer.get_feature_names_out(),
    'Importance': rf_model.feature_importances_
}).sort_values(by='Importance', ascending=False)

print("\nTop features affecting satisfaction:")
print(feature_importance.head(10))

# Extract key improvement areas based on important features and negative sentiment
def extract_improvement_areas(data, feature_importance, top_n=20):
    # Identify samples with lower satisfaction levels
    low_satisfaction = data[data['Satisfaction_Level'].astype(float) < 3.5]
    
    # Get important words from feature importance
    important_words = set(feature_importance.head(top_n)['Feature'])
    
    # Look for these words in low satisfaction feedback
    improvement_areas = {}
    
    for _, row in low_satisfaction.iterrows():
        feedback = row['Processed_Text']
        if isinstance(feedback, str):
            for word in important_words:
                if word in feedback:
                    if word in improvement_areas:
                        improvement_areas[word] += 1
                    else:
                        improvement_areas[word] = 1
    
    return sorted(improvement_areas.items(), key=lambda x: x[1], reverse=True)

improvement_areas = extract_improvement_areas(data, feature_importance)
print("\nKey improvement areas based on analysis:")
for area, count in improvement_areas[:10]:
    print(f"- {area}: {count} mentions in negative feedback")

# Create a suggestions database based on analysis
suggestions_map = {
    'battery': [
        "Increase battery capacity based on user feedback analysis.",
        "Implement smart power management optimized for different usage patterns.",
        "Add battery status indicators with predictive time remaining."
    ],
    'grip': [
        "Redesign ergonomic grip based on hand size clustering from user data.",
        "Implement adjustable grip options for different user preferences."
    ],
    'weight': [
        "Optimize weight distribution based on usage pattern analysis.",
        "Offer lightweight alternatives identified through sentiment patterns."
    ],
    'app': [
        "Enhance app connectivity based on user interaction data.",
        "Add features users frequently request in feedback (smart alerts, nutrition tracking)."
    ],
    'temperature': [
        "Implement adaptive temperature control based on usage patterns.",
        "Add temperature presets derived from positive feedback clusters."
    ],
    'sync': [
        "Improve synchronization reliability based on pain points identified.",
        "Develop auto-reconnect features based on usage pattern analysis."
    ],
    'customizable': [
        "Expand customization options based on feature importance analysis.",
        "Create user-specific presets derived from satisfaction predictors."
    ],
    'quality': [
        "Focus on quality improvements in areas most correlated with satisfaction.",
        "Implement quality control checks for features mentioned in negative clusters."
    ],
    'price': [
        "Analyze price-to-satisfaction ratio for different user segments.",
        "Develop targeted pricing strategies based on satisfaction predictors."
    ],
    'salt': [
        "Implement adaptive salt dispensing based on taste preference clusters.",
        "Develop salt level customization based on sentiment analysis findings."
    ]
}

# Add dynamic generation of suggestions based on analysis
for word, _ in improvement_areas[:5]:
    if word not in suggestions_map and len(word) > 3:  # Filter out short words
        related_words = []
        if word in word2vec_model.wv:
            related_words = [w for w, _ in word2vec_model.wv.most_similar(word, topn=3)]
        
        suggestions_map[word] = [
            f"Improve {word} functionality based on user feedback analysis.",
            f"Consider redesigning {word} features mentioned in negative reviews.",
            f"Research alternative approaches to {word} related features."
        ]

# Main analysis function for user feedback
def analyze_feedback(user_feedback, word2vec_model, tfidf_vectorizer, rf_model, sentiment_analyzer, suggestions_map):
    # Preprocess the feedback
    processed_feedback = preprocess_text(user_feedback)
    
    # Advanced sentiment analysis
    try:
        sentiment_result = sentiment_analyzer(user_feedback)
        sentiment = sentiment_result[0]['label']
        sentiment_score = sentiment_result[0]['score']
    except:
        # Fallback to a simpler approach if transformer fails
        sentiment = "NEUTRAL"
        sentiment_score = 0.5
    
    # Vectorize for prediction
    user_feedback_vec = tfidf_vectorizer.transform([processed_feedback])
    
    # Predict satisfaction level
    predicted_satisfaction = rf_model.predict(user_feedback_vec)[0]
    
    # Get document embedding for similarity analysis
    tokens = processed_feedback.split()
    doc_vec = document_vector(word2vec_model, tokens)
    
    # Generate improvement suggestions based on feedback content
    improvement_suggestions = []
    
    # Check for keywords in processed feedback
    for keyword, suggestions in suggestions_map.items():
        if keyword in processed_feedback:
            # Choose the most relevant suggestion based on word similarity
            if len(suggestions) > 1 and keyword in word2vec_model.wv:
                # Find most similar words to the feedback
                feedback_tokens = set(processed_feedback.split())
                most_similar_idx = 0
                highest_similarity = -1
                
                for i, suggestion in enumerate(suggestions):
                    # Calculate similarity between suggestion and feedback
                    suggestion_tokens = set(preprocess_text(suggestion).split())
                    overlap = feedback_tokens.intersection(suggestion_tokens)
                    similarity = len(overlap) / max(1, len(feedback_tokens.union(suggestion_tokens)))
                    
                    if similarity > highest_similarity:
                        highest_similarity = similarity
                        most_similar_idx = i
                
                improvement_suggestions.append(suggestions[most_similar_idx])
            else:
                improvement_suggestions.append(suggestions[0])
    
    # If no specific keywords were found, generate suggestions based on sentiment
    if not improvement_suggestions:
        if sentiment == "NEGATIVE":
            # Extract important words from the feedback
            feedback_words = set(processed_feedback.split())
            for word in feedback_words:
                if len(word) > 3 and word in feature_importance['Feature'].values:
                    improvement_suggestions.append(f"Consider improving the {word} feature based on user feedback analysis.")
                    break
        
        if not improvement_suggestions:
            improvement_suggestions.append("Thank you for your feedback! Our AI has recorded your comments for future product improvements.")

    # Limit to top 3 most relevant suggestions
    improvement_suggestions = improvement_suggestions[:3]

    return {
        'Sentiment': sentiment,
        'Sentiment_Score': sentiment_score,
        'Predicted_Satisfaction_Level': predicted_satisfaction,
        'Improvement_Suggestions': improvement_suggestions
    }

# Example usage
if __name__ == "__main__":
    print("\n" + "="*50)
    print("Smart Spoon Feedback Analysis System")
    print("="*50)
    
    user_review = input("\nPlease enter your feedback about the Smart Spoon: ")
    
    result = analyze_feedback(
        user_review, 
        word2vec_model, 
        tfidf_vectorizer, 
        rf_model,
        sentiment_analyzer,
        suggestions_map
    )

    # Display the results
    print("\nAnalysis Results:")
    print(f"Sentiment: {result['Sentiment']} (confidence: {result['Sentiment_Score']:.2f})")
    print(f"Predicted Satisfaction Level: {result['Predicted_Satisfaction_Level']}")
    print("\nPersonalized Improvement Suggestions:")
    for suggestion in result['Improvement_Suggestions']:
        print(f"- {suggestion}")
    
    print("\nThank you for helping us improve the Smart Spoon!")
import nltk
nltk.download('punkt_tab')
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.cluster import KMeans
from gensim.models import Word2Vec
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from transformers import pipeline
import warnings
warnings.filterwarnings('ignore')

# Download necessary NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# Load the dataset with appropriate encoding
try:
    data = pd.read_csv('response_data.csv', encoding='utf-8')
except UnicodeDecodeError:
    try:
        data = pd.read_csv('response_data.csv', encoding='ISO-8859-1')
    except UnicodeDecodeError:
        data = pd.read_csv('response_data.csv', encoding='cp1252')

print(f"Loaded dataset with {len(data)} records")

# Data preprocessing function
def preprocess_text(text):
    if isinstance(text, str):
        # Convert to lowercase
        text = text.lower()
        # Remove special characters and numbers
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\d+', '', text)
        # Tokenize
        tokens = nltk.word_tokenize(text)
        # Remove stopwords
        stop_words = set(stopwords.words('english'))
        tokens = [word for word in tokens if word not in stop_words]
        # Lemmatization
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(word) for word in tokens]
        return ' '.join(tokens)
    return ""

# Apply preprocessing to feedback text
print("Preprocessing text data...")
data['Processed_Text'] = data['Feedback_Text'].apply(preprocess_text)

# Create a list of tokenized words for Word2Vec
tokenized_texts = [text.split() for text in data['Processed_Text'] if isinstance(text, str) and text.strip()]

# Word2Vec for word embeddings - capture semantic relationships
print("Training Word2Vec model...")
word2vec_model = Word2Vec(
    tokenized_texts,
    vector_size=100,
    window=5,
    min_count=2,
    workers=4
)

# Function to get document vectors from Word2Vec
def document_vector(word2vec_model, doc):
    # Remove out-of-vocabulary words
    doc = [word for word in doc if word in word2vec_model.wv.key_to_index]
    if len(doc) == 0:
        return np.zeros(word2vec_model.vector_size)
    return np.mean(word2vec_model.wv[doc], axis=0)

# Create document vectors
doc_vectors = []
for tokens in tokenized_texts:
    doc_vectors.append(document_vector(word2vec_model, tokens))

# Create a DataFrame with the vectors for clustering
print("Performing user feedback clustering...")
doc_vectors_df = pd.DataFrame(doc_vectors)

# K-means clustering to identify patterns in feedback
kmeans = KMeans(n_clusters=5, random_state=42)
if len(doc_vectors) > 0:
    data_subset = data.iloc[:len(doc_vectors)].copy()
    data_subset['Cluster'] = kmeans.fit_predict(doc_vectors_df)
    
    # Analyze clusters
    cluster_analysis = data_subset.groupby('Cluster').agg({
        'Satisfaction_Level': 'mean',
        'Feedback_Text': lambda x: ' '.join(x)
    }).reset_index()
    
    # Get top keywords for each cluster
    cluster_analysis['Top_Keywords'] = cluster_analysis['Feedback_Text'].apply(
        lambda x: ' '.join([word for word, count in nltk.FreqDist(nltk.word_tokenize(preprocess_text(x))).most_common(5)])
    )
    
    print("\nCluster Analysis:")
    print(cluster_analysis[['Cluster', 'Satisfaction_Level', 'Top_Keywords']])

# Advanced sentiment analysis using HuggingFace Transformers
print("Performing advanced sentiment analysis...")
sentiment_analyzer = pipeline('sentiment-analysis')

# Apply sentiment analysis to a sample of the data (to avoid rate limits or performance issues)
sample_size = min(100, len(data))
sample_indices = np.random.choice(len(data), sample_size, replace=False)
data['Advanced_Sentiment'] = None

for idx in sample_indices:
    text = data.loc[idx, 'Feedback_Text']
    if isinstance(text, str) and text.strip():
        try:
            result = sentiment_analyzer(text)
            data.loc[idx, 'Advanced_Sentiment'] = result[0]['label']
        except Exception as e:
            print(f"Error analyzing sentiment for index {idx}: {e}")

# TF-IDF Vectorization for better feature extraction
print("Creating TF-IDF features...")
tfidf_vectorizer = TfidfVectorizer(
    max_features=1000,
    min_df=2,
    max_df=0.85
)
tfidf_features = tfidf_vectorizer.fit_transform(data['Processed_Text'].fillna(''))

# Prepare data for satisfaction level prediction
X = tfidf_features
y = data['Satisfaction_Level'].astype(str)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a more advanced RandomForest model for satisfaction prediction
print("Training RandomForest classifier...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Evaluate the model
y_pred = rf_model.predict(X_test)
print("\nModel Evaluation:")
print(classification_report(y_test, y_pred))

# Feature importance analysis
feature_importance = pd.DataFrame({
    'Feature': tfidf_vectorizer.get_feature_names_out(),
    'Importance': rf_model.feature_importances_
}).sort_values(by='Importance', ascending=False)

print("\nTop features affecting satisfaction:")
print(feature_importance.head(10))

# Extract key improvement areas based on important features and negative sentiment
def extract_improvement_areas(data, feature_importance, top_n=20):
    # Identify samples with lower satisfaction levels
    low_satisfaction = data[data['Satisfaction_Level'].astype(float) < 3.5]
    
    # Get important words from feature importance
    important_words = set(feature_importance.head(top_n)['Feature'])
    
    # Look for these words in low satisfaction feedback
    improvement_areas = {}
    
    for _, row in low_satisfaction.iterrows():
        feedback = row['Processed_Text']
        if isinstance(feedback, str):
            for word in important_words:
                if word in feedback:
                    if word in improvement_areas:
                        improvement_areas[word] += 1
                    else:
                        improvement_areas[word] = 1
    
    return sorted(improvement_areas.items(), key=lambda x: x[1], reverse=True)

improvement_areas = extract_improvement_areas(data, feature_importance)
print("\nKey improvement areas based on analysis:")
for area, count in improvement_areas[:10]:
    print(f"- {area}: {count} mentions in negative feedback")

# Create a suggestions database based on analysis
suggestions_map = {
    'battery': [
        "Increase battery capacity based on user feedback analysis.",
        "Implement smart power management optimized for different usage patterns.",
        "Add battery status indicators with predictive time remaining."
    ],
    'grip': [
        "Redesign ergonomic grip based on hand size clustering from user data.",
        "Implement adjustable grip options for different user preferences."
    ],
    'weight': [
        "Optimize weight distribution based on usage pattern analysis.",
        "Offer lightweight alternatives identified through sentiment patterns."
    ],
    'app': [
        "Enhance app connectivity based on user interaction data.",
        "Add features users frequently request in feedback (smart alerts, nutrition tracking)."
    ],
    'temperature': [
        "Implement adaptive temperature control based on usage patterns.",
        "Add temperature presets derived from positive feedback clusters."
    ],
    'sync': [
        "Improve synchronization reliability based on pain points identified.",
        "Develop auto-reconnect features based on usage pattern analysis."
    ],
    'customizable': [
        "Expand customization options based on feature importance analysis.",
        "Create user-specific presets derived from satisfaction predictors."
    ],
    'quality': [
        "Focus on quality improvements in areas most correlated with satisfaction.",
        "Implement quality control checks for features mentioned in negative clusters."
    ],
    'price': [
        "Analyze price-to-satisfaction ratio for different user segments.",
        "Develop targeted pricing strategies based on satisfaction predictors."
    ],
    'salt': [
        "Implement adaptive salt dispensing based on taste preference clusters.",
        "Develop salt level customization based on sentiment analysis findings."
    ]
}

# Add dynamic generation of suggestions based on analysis
for word, _ in improvement_areas[:5]:
    if word not in suggestions_map and len(word) > 3:  # Filter out short words
        related_words = []
        if word in word2vec_model.wv:
            related_words = [w for w, _ in word2vec_model.wv.most_similar(word, topn=3)]
        
        suggestions_map[word] = [
            f"Improve {word} functionality based on user feedback analysis.",
            f"Consider redesigning {word} features mentioned in negative reviews.",
            f"Research alternative approaches to {word} related features."
        ]

# Main analysis function for user feedback
def analyze_feedback(user_feedback, word2vec_model, tfidf_vectorizer, rf_model, sentiment_analyzer, suggestions_map):
    # Preprocess the feedback
    processed_feedback = preprocess_text(user_feedback)
    
    # Advanced sentiment analysis
    try:
        sentiment_result = sentiment_analyzer(user_feedback)
        sentiment = sentiment_result[0]['label']
        sentiment_score = sentiment_result[0]['score']
    except:
        # Fallback to a simpler approach if transformer fails
        sentiment = "NEUTRAL"
        sentiment_score = 0.5
    
    # Vectorize for prediction
    user_feedback_vec = tfidf_vectorizer.transform([processed_feedback])
    
    # Predict satisfaction level
    predicted_satisfaction = rf_model.predict(user_feedback_vec)[0]
    
    # Get document embedding for similarity analysis
    tokens = processed_feedback.split()
    doc_vec = document_vector(word2vec_model, tokens)
    
    # Generate improvement suggestions based on feedback content
    improvement_suggestions = []
    
    # Check for keywords in processed feedback
    for keyword, suggestions in suggestions_map.items():
        if keyword in processed_feedback:
            # Choose the most relevant suggestion based on word similarity
            if len(suggestions) > 1 and keyword in word2vec_model.wv:
                # Find most similar words to the feedback
                feedback_tokens = set(processed_feedback.split())
                most_similar_idx = 0
                highest_similarity = -1
                
                for i, suggestion in enumerate(suggestions):
                    # Calculate similarity between suggestion and feedback
                    suggestion_tokens = set(preprocess_text(suggestion).split())
                    overlap = feedback_tokens.intersection(suggestion_tokens)
                    similarity = len(overlap) / max(1, len(feedback_tokens.union(suggestion_tokens)))
                    
                    if similarity > highest_similarity:
                        highest_similarity = similarity
                        most_similar_idx = i
                
                improvement_suggestions.append(suggestions[most_similar_idx])
            else:
                improvement_suggestions.append(suggestions[0])
    
    # If no specific keywords were found, generate suggestions based on sentiment
    if not improvement_suggestions:
        if sentiment == "NEGATIVE":
            # Extract important words from the feedback
            feedback_words = set(processed_feedback.split())
            for word in feedback_words:
                if len(word) > 3 and word in feature_importance['Feature'].values:
                    improvement_suggestions.append(f"Consider improving the {word} feature based on user feedback analysis.")
                    break
        
        if not improvement_suggestions:
            improvement_suggestions.append("Thank you for your feedback! Our AI has recorded your comments for future product improvements.")

    # Limit to top 3 most relevant suggestions
    improvement_suggestions = improvement_suggestions[:3]

    return {
        'Sentiment': sentiment,
        'Sentiment_Score': sentiment_score,
        'Predicted_Satisfaction_Level': predicted_satisfaction,
        'Improvement_Suggestions': improvement_suggestions
    }

# Example usage
if __name__ == "__main__":
    print("\n" + "="*50)
    print("Smart Spoon Feedback Analysis System")
    print("="*50)
    
    user_review = input("\nPlease enter your feedback about the Smart Spoon: ")
    
    result = analyze_feedback(
        user_review, 
        word2vec_model, 
        tfidf_vectorizer, 
        rf_model,
        sentiment_analyzer,
        suggestions_map
    )

    # Display the results
    print("\nAnalysis Results:")
    print(f"Sentiment: {result['Sentiment']} (confidence: {result['Sentiment_Score']:.2f})")
    print(f"Predicted Satisfaction Level: {result['Predicted_Satisfaction_Level']}")
    print("\nPersonalized Improvement Suggestions:")
    for suggestion in result['Improvement_Suggestions']:
        print(f"- {suggestion}")
    
    print("\nThank you for helping us improve the Smart Spoon!")
pip show tensorflow keras transformers
pip install --upgrade tensorflow keras transformers
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.cluster import KMeans
from gensim.models import Word2Vec
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from transformers import pipeline
import warnings
warnings.filterwarnings('ignore')

# Download necessary NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# Load the dataset with appropriate encoding
try:
    data = pd.read_csv('response_data.csv', encoding='utf-8')
except UnicodeDecodeError:
    try:
        data = pd.read_csv('response_data.csv', encoding='ISO-8859-1')
    except UnicodeDecodeError:
        data = pd.read_csv('response_data.csv', encoding='cp1252')

print(f"Loaded dataset with {len(data)} records")

# Data preprocessing function
def preprocess_text(text):
    if isinstance(text, str):
        # Convert to lowercase
        text = text.lower()
        # Remove special characters and numbers
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\d+', '', text)
        # Tokenize
        tokens = nltk.word_tokenize(text)
        # Remove stopwords
        stop_words = set(stopwords.words('english'))
        tokens = [word for word in tokens if word not in stop_words]
        # Lemmatization
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(word) for word in tokens]
        return ' '.join(tokens)
    return ""

# Apply preprocessing to feedback text
print("Preprocessing text data...")
data['Processed_Text'] = data['Feedback_Text'].apply(preprocess_text)

# Create a list of tokenized words for Word2Vec
tokenized_texts = [text.split() for text in data['Processed_Text'] if isinstance(text, str) and text.strip()]

# Word2Vec for word embeddings - capture semantic relationships
print("Training Word2Vec model...")
word2vec_model = Word2Vec(
    tokenized_texts,
    vector_size=100,
    window=5,
    min_count=2,
    workers=4
)

# Function to get document vectors from Word2Vec
def document_vector(word2vec_model, doc):
    # Remove out-of-vocabulary words
    doc = [word for word in doc if word in word2vec_model.wv.key_to_index]
    if len(doc) == 0:
        return np.zeros(word2vec_model.vector_size)
    return np.mean(word2vec_model.wv[doc], axis=0)

# Create document vectors
doc_vectors = []
for tokens in tokenized_texts:
    doc_vectors.append(document_vector(word2vec_model, tokens))

# Create a DataFrame with the vectors for clustering
print("Performing user feedback clustering...")
doc_vectors_df = pd.DataFrame(doc_vectors)

# K-means clustering to identify patterns in feedback
kmeans = KMeans(n_clusters=5, random_state=42)
if len(doc_vectors) > 0:
    data_subset = data.iloc[:len(doc_vectors)].copy()
    data_subset['Cluster'] = kmeans.fit_predict(doc_vectors_df)
    
    # Analyze clusters
    cluster_analysis = data_subset.groupby('Cluster').agg({
        'Satisfaction_Level': 'mean',
        'Feedback_Text': lambda x: ' '.join(x)
    }).reset_index()
    
    # Get top keywords for each cluster
    cluster_analysis['Top_Keywords'] = cluster_analysis['Feedback_Text'].apply(
        lambda x: ' '.join([word for word, count in nltk.FreqDist(nltk.word_tokenize(preprocess_text(x))).most_common(5)])
    )
    
    print("\nCluster Analysis:")
    print(cluster_analysis[['Cluster', 'Satisfaction_Level', 'Top_Keywords']])

# Advanced sentiment analysis using HuggingFace Transformers
print("Performing advanced sentiment analysis...")
sentiment_analyzer = pipeline('sentiment-analysis')

# Apply sentiment analysis to a sample of the data (to avoid rate limits or performance issues)
sample_size = min(100, len(data))
sample_indices = np.random.choice(len(data), sample_size, replace=False)
data['Advanced_Sentiment'] = None

for idx in sample_indices:
    text = data.loc[idx, 'Feedback_Text']
    if isinstance(text, str) and text.strip():
        try:
            result = sentiment_analyzer(text)
            data.loc[idx, 'Advanced_Sentiment'] = result[0]['label']
        except Exception as e:
            print(f"Error analyzing sentiment for index {idx}: {e}")

# TF-IDF Vectorization for better feature extraction
print("Creating TF-IDF features...")
tfidf_vectorizer = TfidfVectorizer(
    max_features=1000,
    min_df=2,
    max_df=0.85
)
tfidf_features = tfidf_vectorizer.fit_transform(data['Processed_Text'].fillna(''))

# Prepare data for satisfaction level prediction
X = tfidf_features
y = data['Satisfaction_Level'].astype(str)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a more advanced RandomForest model for satisfaction prediction
print("Training RandomForest classifier...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Evaluate the model
y_pred = rf_model.predict(X_test)
print("\nModel Evaluation:")
print(classification_report(y_test, y_pred))

# Feature importance analysis
feature_importance = pd.DataFrame({
    'Feature': tfidf_vectorizer.get_feature_names_out(),
    'Importance': rf_model.feature_importances_
}).sort_values(by='Importance', ascending=False)

print("\nTop features affecting satisfaction:")
print(feature_importance.head(10))

# Extract key improvement areas based on important features and negative sentiment
def extract_improvement_areas(data, feature_importance, top_n=20):
    # Identify samples with lower satisfaction levels
    low_satisfaction = data[data['Satisfaction_Level'].astype(float) < 3.5]
    
    # Get important words from feature importance
    important_words = set(feature_importance.head(top_n)['Feature'])
    
    # Look for these words in low satisfaction feedback
    improvement_areas = {}
    
    for _, row in low_satisfaction.iterrows():
        feedback = row['Processed_Text']
        if isinstance(feedback, str):
            for word in important_words:
                if word in feedback:
                    if word in improvement_areas:
                        improvement_areas[word] += 1
                    else:
                        improvement_areas[word] = 1
    
    return sorted(improvement_areas.items(), key=lambda x: x[1], reverse=True)

improvement_areas = extract_improvement_areas(data, feature_importance)
print("\nKey improvement areas based on analysis:")
for area, count in improvement_areas[:10]:
    print(f"- {area}: {count} mentions in negative feedback")

# Create a suggestions database based on analysis
suggestions_map = {
    'battery': [
        "Increase battery capacity based on user feedback analysis.",
        "Implement smart power management optimized for different usage patterns.",
        "Add battery status indicators with predictive time remaining."
    ],
    'grip': [
        "Redesign ergonomic grip based on hand size clustering from user data.",
        "Implement adjustable grip options for different user preferences."
    ],
    'weight': [
        "Optimize weight distribution based on usage pattern analysis.",
        "Offer lightweight alternatives identified through sentiment patterns."
    ],
    'app': [
        "Enhance app connectivity based on user interaction data.",
        "Add features users frequently request in feedback (smart alerts, nutrition tracking)."
    ],
    'temperature': [
        "Implement adaptive temperature control based on usage patterns.",
        "Add temperature presets derived from positive feedback clusters."
    ],
    'sync': [
        "Improve synchronization reliability based on pain points identified.",
        "Develop auto-reconnect features based on usage pattern analysis."
    ],
    'customizable': [
        "Expand customization options based on feature importance analysis.",
        "Create user-specific presets derived from satisfaction predictors."
    ],
    'quality': [
        "Focus on quality improvements in areas most correlated with satisfaction.",
        "Implement quality control checks for features mentioned in negative clusters."
    ],
    'price': [
        "Analyze price-to-satisfaction ratio for different user segments.",
        "Develop targeted pricing strategies based on satisfaction predictors."
    ],
    'salt': [
        "Implement adaptive salt dispensing based on taste preference clusters.",
        "Develop salt level customization based on sentiment analysis findings."
    ]
}

# Add dynamic generation of suggestions based on analysis
for word, _ in improvement_areas[:5]:
    if word not in suggestions_map and len(word) > 3:  # Filter out short words
        related_words = []
        if word in word2vec_model.wv:
            related_words = [w for w, _ in word2vec_model.wv.most_similar(word, topn=3)]
        
        suggestions_map[word] = [
            f"Improve {word} functionality based on user feedback analysis.",
            f"Consider redesigning {word} features mentioned in negative reviews.",
            f"Research alternative approaches to {word} related features."
        ]

# Main analysis function for user feedback
def analyze_feedback(user_feedback, word2vec_model, tfidf_vectorizer, rf_model, sentiment_analyzer, suggestions_map):
    # Preprocess the feedback
    processed_feedback = preprocess_text(user_feedback)
    
    # Advanced sentiment analysis
    try:
        sentiment_result = sentiment_analyzer(user_feedback)
        sentiment = sentiment_result[0]['label']
        sentiment_score = sentiment_result[0]['score']
    except:
        # Fallback to a simpler approach if transformer fails
        sentiment = "NEUTRAL"
        sentiment_score = 0.5
    
    # Vectorize for prediction
    user_feedback_vec = tfidf_vectorizer.transform([processed_feedback])
    
    # Predict satisfaction level
    predicted_satisfaction = rf_model.predict(user_feedback_vec)[0]
    
    # Get document embedding for similarity analysis
    tokens = processed_feedback.split()
    doc_vec = document_vector(word2vec_model, tokens)
    
    # Generate improvement suggestions based on feedback content
    improvement_suggestions = []
    
    # Check for keywords in processed feedback
    for keyword, suggestions in suggestions_map.items():
        if keyword in processed_feedback:
            # Choose the most relevant suggestion based on word similarity
            if len(suggestions) > 1 and keyword in word2vec_model.wv:
                # Find most similar words to the feedback
                feedback_tokens = set(processed_feedback.split())
                most_similar_idx = 0
                highest_similarity = -1
                
                for i, suggestion in enumerate(suggestions):
                    # Calculate similarity between suggestion and feedback
                    suggestion_tokens = set(preprocess_text(suggestion).split())
                    overlap = feedback_tokens.intersection(suggestion_tokens)
                    similarity = len(overlap) / max(1, len(feedback_tokens.union(suggestion_tokens)))
                    
                    if similarity > highest_similarity:
                        highest_similarity = similarity
                        most_similar_idx = i
                
                improvement_suggestions.append(suggestions[most_similar_idx])
            else:
                improvement_suggestions.append(suggestions[0])
    
    # If no specific keywords were found, generate suggestions based on sentiment
    if not improvement_suggestions:
        if sentiment == "NEGATIVE":
            # Extract important words from the feedback
            feedback_words = set(processed_feedback.split())
            for word in feedback_words:
                if len(word) > 3 and word in feature_importance['Feature'].values:
                    improvement_suggestions.append(f"Consider improving the {word} feature based on user feedback analysis.")
                    break
        
        if not improvement_suggestions:
            improvement_suggestions.append("Thank you for your feedback! Our AI has recorded your comments for future product improvements.")

    # Limit to top 3 most relevant suggestions
    improvement_suggestions = improvement_suggestions[:3]

    return {
        'Sentiment': sentiment,
        'Sentiment_Score': sentiment_score,
        'Predicted_Satisfaction_Level': predicted_satisfaction,
        'Improvement_Suggestions': improvement_suggestions
    }

# Example usage
if __name__ == "__main__":
    print("\n" + "="*50)
    print("Smart Spoon Feedback Analysis System")
    print("="*50)
    
    user_review = input("\nPlease enter your feedback about the Smart Spoon: ")
    
    result = analyze_feedback(
        user_review, 
        word2vec_model, 
        tfidf_vectorizer, 
        rf_model,
        sentiment_analyzer,
        suggestions_map
    )

    # Display the results
    print("\nAnalysis Results:")
    print(f"Sentiment: {result['Sentiment']} (confidence: {result['Sentiment_Score']:.2f})")
    print(f"Predicted Satisfaction Level: {result['Predicted_Satisfaction_Level']}")
    print("\nPersonalized Improvement Suggestions:")
    for suggestion in result['Improvement_Suggestions']:
        print(f"- {suggestion}")
    
    print("\nThank you for helping us improve the Smart Spoon!")
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.cluster import KMeans
from gensim.models import Word2Vec
import matplotlib.pyplot as plt
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Download necessary NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# Load the dataset with appropriate encoding
try:
    data = pd.read_csv('response_data.csv', encoding='utf-8')
except UnicodeDecodeError:
    try:
        data = pd.read_csv('response_data.csv', encoding='ISO-8859-1')
    except UnicodeDecodeError:
        data = pd.read_csv('response_data.csv', encoding='cp1252')

print(f"Loaded dataset with {len(data)} records")

# Data preprocessing function
def preprocess_text(text):
    if isinstance(text, str):
        # Convert to lowercase
        text = text.lower()
        # Remove special characters and numbers
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\d+', '', text)
        # Tokenize
        tokens = nltk.word_tokenize(text)
        # Remove stopwords
        stop_words = set(stopwords.words('english'))
        tokens = [word for word in tokens if word not in stop_words]
        # Lemmatization
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(word) for word in tokens]
        return ' '.join(tokens)
    return ""

# Apply preprocessing to feedback text
print("Preprocessing text data...")
data['Processed_Text'] = data['Feedback_Text'].apply(preprocess_text)

# Create a list of tokenized words for Word2Vec
tokenized_texts = [text.split() for text in data['Processed_Text'] if isinstance(text, str) and text.strip()]

# Word2Vec for word embeddings - capture semantic relationships
print("Training Word2Vec model...")
word2vec_model = Word2Vec(
    tokenized_texts,
    vector_size=100,
    window=5,
    min_count=2,
    workers=4
)

# Function to get document vectors from Word2Vec
def document_vector(word2vec_model, doc):
    # Remove out-of-vocabulary words
    doc = [word for word in doc if word in word2vec_model.wv.key_to_index]
    if len(doc) == 0:
        return np.zeros(word2vec_model.vector_size)
    return np.mean(word2vec_model.wv[doc], axis=0)

# Create document vectors
doc_vectors = []
for tokens in tokenized_texts:
    doc_vectors.append(document_vector(word2vec_model, tokens))

# Create a DataFrame with the vectors for clustering
print("Performing user feedback clustering...")
doc_vectors_df = pd.DataFrame(doc_vectors)

# K-means clustering to identify patterns in feedback
kmeans = KMeans(n_clusters=5, random_state=42)
if len(doc_vectors) > 0:
    data_subset = data.iloc[:len(doc_vectors)].copy()
    data_subset['Cluster'] = kmeans.fit_predict(doc_vectors_df)
    
    # Analyze clusters
    cluster_analysis = data_subset.groupby('Cluster').agg({
        'Satisfaction_Level': 'mean',
        'Feedback_Text': lambda x: ' '.join(x)
    }).reset_index()
    
    # Get top keywords for each cluster
    cluster_analysis['Top_Keywords'] = cluster_analysis['Feedback_Text'].apply(
        lambda x: ' '.join([word for word, count in nltk.FreqDist(nltk.word_tokenize(preprocess_text(x))).most_common(5)])
    )
    
    print("\nCluster Analysis:")
    print(cluster_analysis[['Cluster', 'Satisfaction_Level', 'Top_Keywords']])

# Enhanced sentiment analysis using VADER with context features
print("Performing enhanced sentiment analysis...")
sid = SentimentIntensityAnalyzer()

# Create additional NLP features for context-aware sentiment
data['Sentiment_Compound'] = data['Feedback_Text'].apply(
    lambda x: sid.polarity_scores(x)['compound'] if isinstance(x, str) else 0
)
data['Sentiment_Positive'] = data['Feedback_Text'].apply(
    lambda x: sid.polarity_scores(x)['pos'] if isinstance(x, str) else 0
)
data['Sentiment_Negative'] = data['Feedback_Text'].apply(
    lambda x: sid.polarity_scores(x)['neg'] if isinstance(x, str) else 0
)
data['Sentiment_Neutral'] = data['Feedback_Text'].apply(
    lambda x: sid.polarity_scores(x)['neu'] if isinstance(x, str) else 0
)

# Create a more refined sentiment category
data['Advanced_Sentiment'] = pd.cut(
    data['Sentiment_Compound'], 
    bins=[-1, -0.5, -0.1, 0.1, 0.5, 1], 
    labels=['Very Negative', 'Negative', 'Neutral', 'Positive', 'Very Positive']
)

# Analyze sentiment distribution
sentiment_distribution = data['Advanced_Sentiment'].value_counts()
print("\nSentiment Distribution:")
for sentiment, count in sentiment_distribution.items():
    print(f"- {sentiment}: {count} ({count/len(data)*100:.1f}%)")

# Extract key phrases for each sentiment category
def extract_key_phrases(text_series, top_n=5):
    # Extract bigrams and trigrams that appear frequently
    all_text = ' '.join(text_series.fillna(''))
    tokens = nltk.word_tokenize(all_text)
    bigrams = list(nltk.bigrams(tokens))
    trigrams = list(nltk.trigrams(tokens))
    
    # Convert to strings for easier counting
    bigram_phrases = [' '.join(bg) for bg in bigrams]
    trigram_phrases = [' '.join(tg) for tg in trigrams]
    
    # Count and return top phrases
    all_phrases = bigram_phrases + trigram_phrases
    most_common = Counter(all_phrases).most_common(top_n)
    return [phrase for phrase, count in most_common]

# Get key phrases by sentiment
print("\nKey phrases by sentiment:")
for sentiment in data['Advanced_Sentiment'].unique():
    if pd.notna(sentiment):
        sentiment_texts = data[data['Advanced_Sentiment'] == sentiment]['Processed_Text']
        if len(sentiment_texts) > 0:
            key_phrases = extract_key_phrases(sentiment_texts)
            print(f"- {sentiment}: {', '.join(key_phrases)}")

# TF-IDF Vectorization for better feature extraction
print("Creating TF-IDF features...")
tfidf_vectorizer = TfidfVectorizer(
    max_features=1000,
    min_df=2,
    max_df=0.85
)
tfidf_features = tfidf_vectorizer.fit_transform(data['Processed_Text'].fillna(''))

# Prepare data for satisfaction level prediction
X = tfidf_features
y = data['Satisfaction_Level'].astype(str)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a more advanced RandomForest model for satisfaction prediction
print("Training RandomForest classifier...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Evaluate the model
y_pred = rf_model.predict(X_test)
print("\nModel Evaluation:")
print(classification_report(y_test, y_pred))

# Feature importance analysis
feature_importance = pd.DataFrame({
    'Feature': tfidf_vectorizer.get_feature_names_out(),
    'Importance': rf_model.feature_importances_
}).sort_values(by='Importance', ascending=False)

print("\nTop features affecting satisfaction:")
print(feature_importance.head(10))

# Extract key improvement areas based on important features and negative sentiment
def extract_improvement_areas(data, feature_importance, top_n=20):
    # Identify samples with lower satisfaction levels
    low_satisfaction = data[data['Satisfaction_Level'].astype(float) < 3.5]
    
    # Get important words from feature importance
    important_words = set(feature_importance.head(top_n)['Feature'])
    
    # Look for these words in low satisfaction feedback
    improvement_areas = {}
    
    for _, row in low_satisfaction.iterrows():
        feedback = row['Processed_Text']
        if isinstance(feedback, str):
            for word in important_words:
                if word in feedback:
                    if word in improvement_areas:
                        improvement_areas[word] += 1
                    else:
                        improvement_areas[word] = 1
    
    return sorted(improvement_areas.items(), key=lambda x: x[1], reverse=True)

improvement_areas = extract_improvement_areas(data, feature_importance)
print("\nKey improvement areas based on analysis:")
for area, count in improvement_areas[:10]:
    print(f"- {area}: {count} mentions in negative feedback")

# Create a suggestions database based on analysis
suggestions_map = {
    'battery': [
        "Increase battery capacity based on user feedback analysis.",
        "Implement smart power management optimized for different usage patterns.",
        "Add battery status indicators with predictive time remaining."
    ],
    'grip': [
        "Redesign ergonomic grip based on hand size clustering from user data.",
        "Implement adjustable grip options for different user preferences."
    ],
    'weight': [
        "Optimize weight distribution based on usage pattern analysis.",
        "Offer lightweight alternatives identified through sentiment patterns."
    ],
    'app': [
        "Enhance app connectivity based on user interaction data.",
        "Add features users frequently request in feedback (smart alerts, nutrition tracking)."
    ],
    'temperature': [
        "Implement adaptive temperature control based on usage patterns.",
        "Add temperature presets derived from positive feedback clusters."
    ],
    'sync': [
        "Improve synchronization reliability based on pain points identified.",
        "Develop auto-reconnect features based on usage pattern analysis."
    ],
    'customizable': [
        "Expand customization options based on feature importance analysis.",
        "Create user-specific presets derived from satisfaction predictors."
    ],
    'quality': [
        "Focus on quality improvements in areas most correlated with satisfaction.",
        "Implement quality control checks for features mentioned in negative clusters."
    ],
    'price': [
        "Analyze price-to-satisfaction ratio for different user segments.",
        "Develop targeted pricing strategies based on satisfaction predictors."
    ],
    'salt': [
        "Implement adaptive salt dispensing based on taste preference clusters.",
        "Develop salt level customization based on sentiment analysis findings."
    ]
}

# Add dynamic generation of suggestions based on analysis
for word, _ in improvement_areas[:5]:
    if word not in suggestions_map and len(word) > 3:  # Filter out short words
        related_words = []
        if word in word2vec_model.wv:
            related_words = [w for w, _ in word2vec_model.wv.most_similar(word, topn=3)]
        
        suggestions_map[word] = [
            f"Improve {word} functionality based on user feedback analysis.",
            f"Consider redesigning {word} features mentioned in negative reviews.",
            f"Research alternative approaches to {word} related features."
        ]

# Main analysis function for user feedback
def analyze_feedback(user_feedback, word2vec_model, tfidf_vectorizer, rf_model, sentiment_analyzer, suggestions_map):
    # Preprocess the feedback
    processed_feedback = preprocess_text(user_feedback)
    
    # Advanced sentiment analysis
    try:
        sentiment_result = sentiment_analyzer(user_feedback)
        sentiment = sentiment_result[0]['label']
        sentiment_score = sentiment_result[0]['score']
    except:
        # Fallback to a simpler approach if transformer fails
        sentiment = "NEUTRAL"
        sentiment_score = 0.5
    
    # Vectorize for prediction
    user_feedback_vec = tfidf_vectorizer.transform([processed_feedback])
    
    # Predict satisfaction level
    predicted_satisfaction = rf_model.predict(user_feedback_vec)[0]
    
    # Get document embedding for similarity analysis
    tokens = processed_feedback.split()
    doc_vec = document_vector(word2vec_model, tokens)
    
    # Generate improvement suggestions based on feedback content
    improvement_suggestions = []
    
    # Check for keywords in processed feedback
    for keyword, suggestions in suggestions_map.items():
        if keyword in processed_feedback:
            # Choose the most relevant suggestion based on word similarity
            if len(suggestions) > 1 and keyword in word2vec_model.wv:
                # Find most similar words to the feedback
                feedback_tokens = set(processed_feedback.split())
                most_similar_idx = 0
                highest_similarity = -1
                
                for i, suggestion in enumerate(suggestions):
                    # Calculate similarity between suggestion and feedback
                    suggestion_tokens = set(preprocess_text(suggestion).split())
                    overlap = feedback_tokens.intersection(suggestion_tokens)
                    similarity = len(overlap) / max(1, len(feedback_tokens.union(suggestion_tokens)))
                    
                    if similarity > highest_similarity:
                        highest_similarity = similarity
                        most_similar_idx = i
                
                improvement_suggestions.append(suggestions[most_similar_idx])
            else:
                improvement_suggestions.append(suggestions[0])
    
    # If no specific keywords were found, generate suggestions based on sentiment
    if not improvement_suggestions:
        if sentiment == "NEGATIVE":
            # Extract important words from the feedback
            feedback_words = set(processed_feedback.split())
            for word in feedback_words:
                if len(word) > 3 and word in feature_importance['Feature'].values:
                    improvement_suggestions.append(f"Consider improving the {word} feature based on user feedback analysis.")
                    break
        
        if not improvement_suggestions:
            improvement_suggestions.append("Thank you for your feedback! Our AI has recorded your comments for future product improvements.")

    # Limit to top 3 most relevant suggestions
    improvement_suggestions = improvement_suggestions[:3]

    return {
        'Sentiment': sentiment,
        'Sentiment_Score': sentiment_score,
        'Predicted_Satisfaction_Level': predicted_satisfaction,
        'Improvement_Suggestions': improvement_suggestions
    }

# Example usage
if __name__ == "__main__":
    print("\n" + "="*50)
    print("Smart Spoon Feedback Analysis System")
    print("="*50)
    
    user_review = input("\nPlease enter your feedback about the Smart Spoon: ")
    
    result = analyze_feedback(
        user_review, 
        word2vec_model, 
        tfidf_vectorizer, 
        rf_model,
        sentiment_analyzer,
        suggestions_map
    )

    # Display the results
    print("\nAnalysis Results:")
    print(f"Sentiment: {result['Sentiment']} (confidence: {result['Sentiment_Score']:.2f})")
    print(f"Predicted Satisfaction Level: {result['Predicted_Satisfaction_Level']}")
    print("\nPersonalized Improvement Suggestions:")
    for suggestion in result['Improvement_Suggestions']:
        print(f"- {suggestion}")
    
    print("\nThank you for helping us improve the Smart Spoon!")
%history -f main1.py
%history -f main1.py
