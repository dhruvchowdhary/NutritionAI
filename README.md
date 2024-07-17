# NutritionAI
<img src="https://github.com/dhruvchowdhary/NutritionAI/blob/main/NutritionAI.jpeg?raw=true" width="170">
NutritionAI was created during HackMIT 2023 with a 36-hour constraint. Contributors: Dhruv Chowdhary, Mridu Prashanth, Ashok Saravanan, and Amish Sethi.

[https://devpost.com/software/remora-stay-hydrated](https://devpost.com/software/nutritional)

## Inspiration
Our inspiration for this project stems from a variety of sources, all converging to create a powerful drive for innovation. Listening to Armen from the MIT-IBM Watson AI Lab, we were struck by the formidable challenges of making real-time predictions. This inspired us to embark on a journey to simplify and enhance the accuracy of an essential aspect of daily life: calorie counting.

As a group of college students deeply invested in personal health, we empathize with the universal need to monitor and manage calorie intake. Beyond just calories, we recognize the importance of tracking other vital nutritional elements, such as protein and other macronutrients, in the foods we consume. Our motivation also extends to fostering an informed awareness of daily nutrient consumption, ultimately encouraging healthier eating habits in people. Our app is designed not only to make life easier but to empower individuals to make healthier food choices effortlessly.

## What it does
<img src="https://github.com/dhruvchowdhary/NutritionAI/blob/main/assets/images/1721198941523-640bdb70-3cc6-47d6-9e43-328c4d59b9b6_2.jpg?raw=true" width="300">
The app enables users to capture a photo of a food item, and in real-time, it provides precise information about its calorie and protein content.

To achieve this, our app goes beyond simple image classification. It utilizes the user's photograph to determine both the identity (e.g., apple) and density of the food item. Additionally, it calculates the volume of the food. Using this data, the app combines the density information (calories/gram or protein/gram) with the estimated mass to deliver accurate calorie and protein estimates for the specific food item photographed!

## How we built it

Flask (machine learning models), React Native (front end), OpenCV (camera)

## Challenges we ran into

Although we could figure out React and OpenCV implementations of recording footage through a device, it was hard to find a React Native (with expo) package that allowed us to take a picture in real-time and live stream to the back end. We solved this by writing a volume estimator that worked on a single image input, and also finding an expo package that worked for our react native usage.

## Accomplishments that we're proud of

Putting this hack together in less than 24 hours!

We first met here 23 hours ago, coming all across the country, representing Purdue, Penn, and Berkeley. With some ideas and various skill sets, we were eager to build a project and decided to team up. We faced substantial hurdles along the way, from navigating the intricate intricacies of training the model to wrestling with the complexities of getting React Native to function seamlessly. Despite these formidable roadblocks, we remained steadfast in our commitment to our original concept, refusing to yield to adversity. Our determination and tenacity ultimately propelled us to overcome challenges that would have daunted many, reinforcing our sense of pride in what we've achieved.

## What we learned

Our journey taught us valuable new skills, primarily in React Native. Some of us gained proficiency in building a basic app using React Native, along with mastering the utilization of the React Native Expo Camera package.

## What's next for NutritionAl

The future for NutritionAl holds exciting possibilities. We envision expanding the app's capabilities to analyze cooked dishes with multiple ingredients. Users will be able to provide a list of ingredients, allowing us to estimate the mass of the dish and calculate its nutritional breakdown by aggregating the nutritional information of each ingredient.

## Other Resources
Our Volume Calculator: [https://github.com/dhruvchowdhary/VolumeCalculator](https://github.com/dhruvchowdhary/VolumeCalculator)
