from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

"""
file_1 = open("Text Files/abstract_1_test", 'r')
corpus_1 = ["USB camera is currently used in daily life for various purposes.",
            "On its development, the use of USB camera can be used to create camera traps and can be used to observe "
            "the development of animal with integrated systems.",
            "In this research, motion detection was used to observe animals online using Single Board Computer (SBC)."
            "Camera trap in this research using Single Board camera in form of raspberry pi 3 B.",
            "Python programming language is used with OpenCV library.",
            "The method used to detect motion is the Mixture of Gaussian (MOG).",
            "The result image gained by motion detection will be uploaded to the dropbox API.",
            "The test performed on 11 videos, the system can process images with 320x240 resolution.",
            "The test results show the best but value of k = 13,"
            "the best threshold value is 100 pixel with an accuracy "
            "of 80,3%, and the maximum distance system can detect animal objects as far as 6m.",
            "The response time gained for the system to process frame per second have average of 0,098 seconds, "
            "while for uploading image to dropbox has an average of 1,618 seconds.",
            "The test result show the system still has room for development and improvement."]


corpus_1_singleline = [
    "USB camera is currently used in daily life for various purposes. On its development, the use of USB camera can be used to create camera traps and can be used to observe the development of animal with integrated systems. In this research, motion detection was used to observe animals online using Single Board Computer (SBC). Camera trap in this research using Single Board camera in form of raspberry pi 3 B. Python programming language is used with OpenCV library. The method used to detect motion is the Mixture of Gaussian (MOG). The result image gained by motion detection will be uploaded to the dropbox API. The test performed on 11 videos, the system can process images with 320x240 resolution. The test results show the best but value of k = 13, the best threshold value is 100 pixel with an accuracy of 80,3%, and the maximum distance system can detect animal objects as far as 6m. The response time gained for the system to process frame per second have average of 0,098 seconds, while for uploading image to dropbox has an average of 1,618 seconds. The test result show the system still has room for development and improvement."]
corpus_2_singleline = [
    "One approach that is often used in forecasting is artificial neural networks (ANN), but ANNs have problems in determining the initial weight value between connections, a long time to reach convergent, and minimum local problems. Deep Belief Network (DBN) model is proposed to improve ANN's ability to forecast exchange rates. DBN is composed of a Restricted Boltzmann Machine (RBM) stack. The DBN structure is optimally determined through experiments. The Adam method is applied to accelerate learning in DBN because it is able to achieve good results quickly compared to other stochastic optimization methods such as Stochastic Gradient Descent (SGD) by maintaining the level of learning for each parameter. Tests are carried out on USD / IDR daily exchange rate data and four evaluation criteria are adopted to evaluate the performance of the proposed method. The DBN-Adam model produces RMSE 59.0635004, MAE 46.406739, MAPE 0.34652. DBN-Adam is also able to reach the point of convergence quickly, where this result is able to outperform the DBN-SGD model."]
corpus_3_singleline = [
    "The server is a computer program or a device that provides functionality for other programs or devices, called \"clients\". Generally, server computers have many resources that can be used by one or more clients through the network with specific permissions and requirements. Therefore, the server needs a monitoring system that can monitor server activity and notify if problems occur. This research focuses on developing a notification and question and answer system to connect the network admin with the monitoring system via chatbot. The developed chatbot can send notifications to the admin if an error occurs and can answer questions about the server's condition. The question and answer system developed implements natural language processing for Indonesian. The process of understanding questions is by classifying each word (token) based on language knowledge stored in the ontology. Then the classification results are processed by rule-base to produce conclusions to take monitoring data and compiled into answers. The test results show that the developed system can auto-notify if any problem in a server, and can answer questions by accuracy 95%."]
"""
corpus_4_singleline = [
    "USB camera is currently used in daily life for various purposes. On its development, the use of USB camera can be used to create camera traps and can be used to observe the development of animal with integrated systems. In this research, motion detection was used to observe animals online using Single Board Computer (SBC). Camera trap in this research using Single Board camera in form of raspberry pi 3 B. Python programming language is used with OpenCV library. The method used to detect motion is the Mixture of Gaussian (MOG). The result image gained by motion detection will be uploaded to the dropbox API. The test performed on 11 videos, the system can process images with 320x240 resolution. The test results show the best but value of k = 13, the best threshold value is 100 pixel with an accuracy of 80,3%, and the maximum distance system can detect animal objects as far as 6m. The response time gained for the system to process frame per second have average of 0,098 seconds, while for uploading image to dropbox has an average of 1,618 seconds. The test result show the system still has room for development and improvement.",
    "One approach that is often used in forecasting is artificial neural networks (ANN), but ANNs have problems in determining the initial weight value between connections, a long time to reach convergent, and minimum local problems. Deep Belief Network (DBN) model is proposed to improve ANN's ability to forecast exchange rates. DBN is composed of a Restricted Boltzmann Machine (RBM) stack. The DBN structure is optimally determined through experiments. The Adam method is applied to accelerate learning in DBN because it is able to achieve good results quickly compared to other stochastic optimization methods such as Stochastic Gradient Descent (SGD) by maintaining the level of learning for each parameter. Tests are carried out on USD / IDR daily exchange rate data and four evaluation criteria are adopted to evaluate the performance of the proposed method. The DBN-Adam model produces RMSE 59.0635004, MAE 46.406739, MAPE 0.34652. DBN-Adam is also able to reach the point of convergence quickly, where this result is able to outperform the DBN-SGD model.",
    "The server is a computer program or a device that provides functionality for other programs or devices, called \"clients\". Generally, server computers have many resources that can be used by one or more clients through the network with specific permissions and requirements. Therefore, the server needs a monitoring system that can monitor server activity and notify if problems occur. This research focuses on developing a notification and question and answer system to connect the network admin with the monitoring system via chatbot. The developed chatbot can send notifications to the admin if an error occurs and can answer questions about the server's condition. The question and answer system developed implements natural language processing for Indonesian. The process of understanding questions is by classifying each word (token) based on language knowledge stored in the ontology. Then the classification results are processed by rule-base to produce conclusions to take monitoring data and compiled into answers. The test results show that the developed system can auto-notify if any problem in a server, and can answer questions by accuracy 95%."
]

vector1 = TfidfVectorizer(stop_words='english')
vector2 = TfidfVectorizer(stop_words='english')
vector3 = TfidfVectorizer(stop_words='english')
vector4 = TfidfVectorizer(stop_words='english')

response4 = vector4.fit_transform(corpus_4_singleline)
print(response4)
print(response4.todense())

df4 = pd.DataFrame(response4.todense().T,
                   index=vector4.get_feature_names_out(corpus_4_singleline),
                   columns=[f'D{i + 1}' for i in range(len(corpus_4_singleline))])
print(df4)


"""
response1 = vector1.fit_transform(corpus_1_singleline)
response2 = vector2.fit_transform(corpus_2_singleline)
response3 = vector3.fit_transform(corpus_3_singleline)


print(response1)
print("\n", response1.todense())

print("\n", response2)
print("\n", response2.todense())

print("\n", response3)
print("\n", response3.todense())
print("\n")

df1 = pd.DataFrame(response1.todense().T,
                   index=vector1.get_feature_names_out(corpus_1_singleline),
                   columns=[f'D{i + 1}' for i in range(len(corpus_1_singleline))])
print(df1)

df2 = pd.DataFrame(response2.todense().T,
                   index=vector2.get_feature_names_out(corpus_2_singleline),
                   columns=[f'D{i + 1}' for i in range(len(corpus_2_singleline))])
print(df2)

df3 = pd.DataFrame(response3.todense().T,
                   index=vector3.get_feature_names_out(corpus_3_singleline),
                   columns=[f'D{i + 1}' for i in range(len(corpus_3_singleline))])
print(df3)
"""