# Author_Classifier
The goal of this project was to be able to use NLP appoaches to train a model to identify the author of a given text.


# Data
I read in 8 different authors and three of their works. The works I used were not limited to just novels. I included short story collections, poems, and essays.
the works I included are as follows:

    -Ernest Hemingway:
     For Whom The Bell Tolls, A Farewell To Arms, The Old Man and The Sea
    -H. G. Wells:
     War of the Worlds, The Invisible Man, The Island of Doctor Moreau
    -Fyodor Dostoefsky 
     Brothers Karmazov, Demons, Crime and Punishment
    -Jane Austen 
     Emma, Pride and Prejudice, Sense and Sensibility 
    -Leo Tolstoy 
     War and Peace, Anna Karenina, What Men Live by and Other Stories 
    -Lewis Carroll 
     Alice in Wonderland, Through the looking glass, Phantasmagoria
    -Mark Twain 
     Tom Sawyer, Huckleberry Fin, The Prince and The Pauper
    -Oscar Wilde 
     The Picture of Dorian Gray, De Profoundis, The Happy Prince

# Methedology

Step 1: I read in the text files and tokenized them by sentence. 

Step 2: I created a chunking function to control the number of sentences in each sample text. I tested 200, 150, 100, 50, 10 and the optimal output was when the sentences were chunked in groups of 100.

Step 3: Tested a baseline with two authors on KNN, LogisitcRegression, DecisionTreeClassifier, and RandomForestClassifier. The RandomForestClassifier with only nop optimized parameters gave me the best output score so I decided to go foward using that one. 

Step 4: I used a Micro F1_Score (Micro F1-score is defined as the harmonic mean of the precision and recall). I decided that this was best because there was no imbalance in categorizing. 

Step 5 I took time optomizing my model and returning the best output possible. 
      
      2 Authors, Training 1: Test 1 Books, Chunk 30
        Seen F1_Score: 100%
        Unseen F1_Score: 86.5%
        
      8 Authors, Training 1: Test 1 Books, Chunk 30
        Seen F1_Score: 90.9%
        Unseen F1_Score: 59.2%
        
      8 Authors, Training 1: Test 1 Books, Chunk 100
        Seen F1_Score: 95.9%
        Unseen F1_Score: 61.7%
        
      8 Authors, Training 2: Test 1 Books, Chunk 100
        Seen F1_Score: 94.9%
        Unseen F1_Score: 66.9%
        
      8 Authors, Training 2: Test 1 Books, Chunk 100, Op. Params(using grid search)
        Seen F1_Score: 94.0%
        Unseen F1_Score: 70.2%
        
# Use Cases

Plagiarism: This could be used to see if a student that turns in a paper is the author of the paper they are turning in. Or even to see if the paper is possibly authored by another student. 

Linguistic Forensics: See if the suspect is author of a given text. To act as further evidence in a case. 

Historical Research: Another use for this model is to be able to see if new found writing samples are from a famous author or not. Maybe to finally to answer the question of if Shakespear is one person or many.

Mimic an Author's Style: It could be used to build a test recommender to help someone write in a specific authors unique style. 


# Further

Given more time I would have like to have been able to use a gradient boosting approach and I hypothesis that I would be able to get an even better F1_Score. I would also like to be able to feed my model even more data and create some way of randomizing the books that are assigned to training and testing data I hypothesis I would get an even better outcome. 







     
