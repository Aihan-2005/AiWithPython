import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import threading
import time


mnist = input_data.read_data_sets('/tmp/mnist_data',one_hot=True)

learning_rate = 0.01
training_epochs = 10
batch_size = 100
n_classes = 10
n_features = 784


cluster_spec = tf.train.ClusterSpec({
    'worker':[
        'localhost:2222',
        'localhost:2223',
        'localhost:2224'
    ]
})


def create_model(x,scope_name):
    with tf.variable_scope(scope_name):
        
        W1 = tf.get_variable('W1',[n_features,256],
                             initializer=tf.random_normal_initailizer(stddev=0.1))
        b1 = tf.get_variable('b1',[256],
                             initializer=tf.constant_initializer(0.1))
        
        layer1 = tf.nn.relu(tf.matmul(x,W1)+b1)
        
        
        W2 = tf.get_variable('W2',[256,128],
                             initializer=tf.random_normal_initializer(stddev=0.1))
        
        b2 = tf.get_variable('b2',[128],
                             initializer=tf.constant_initilizer(0.1))
        
        layer2 = tf.nn.relu(tf.matmul(layer1,W2)+b2)
        
        
        W3 = tf.get_variable('W3',[128,n_classes],
                             initializer=tf.random_normal_initializer(stddev=0.1))
        
        b3 = tf.get_variable('b3',[n_classes],
                             initilizer=tf.constant_initilizer(0.1))
        
        logits = tf.matmul(layer2,W3) + b3
        
    return logits



graph = tf.Graph()

with graph.as_default():
    
    x = tf.placeholder(tf.float32,[None,n_features],name='x')
    y = tf.placeholder(tf.float32,[None,n_classes],name='y')
    
    
    with tf.device('/job:worker/task:0'):
        logits_A = create_model(x,'model_A')
        loss_A = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits_v2(labels=y,logits=logits_A)
        )
        optimizer_A = tf.train.GradientDescentOptimizer(learning_rate)
        train_op_A = optimizer_A.minimize(loss_A)
        predictions_A = tf.nn.softmax(logits_A)
        
        
    with tf.device('/job:worker/task:1'):
        logits_B = create_model(x,'model_B')
        loss_B = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits_v2(labels=y,logits = logits_B)
        )
        optimizer_B = tf.train.AdamOptimizer(learning_rate)
        train_op_B = optimizer_B.minimize(loss_B)
        predictions_B = tf.nn.softmax(logits_B)
        
        
    with tf.device('/job:worker/task:2'):
        logits_C = create_model(x,'model_C')
        loss_C = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits_v2(labels=y,logits = logits_C)
        )
        optimizer_C = tf.train.AdamOptimizer(learning_rate)
        train_op_C = optimizer_C.minimize(loss_C)
        predictions_C = tf.nn.softmax(logits_C)
        
        
    ensemble_predictions = (predictions_A+predictions_B+predictions_C)/3.0
    ensemble_correct = tf.equal(tf.argmax(ensemble_predictions,1),tf.argmax(y,1))
    ensemble_accuracy = tf.reduce_mean(tf.cast(ensemble_correct,tf.float32))
    
    init_op = tf.global_variables_initializer()
    
    
def train_model(sess,train_op,loss,model_name,lock):
    print(f'start learning{model_name}...')
    n_batches = mnist.train.num_examples//batch_size
    
    
    for epoch in range(training_epochs):
        avg_loss = 0.0
        for i in range(n_batches):
            batch_x , batch_y = mnist.train.next_batch(batch_size)
            _,c = sess.run([train_op,loss],feed_dict={x:batch_x , y:batch_y})
            avg_loss += c/n_batches
            
        
        with lock:
            print(f'{model_name}-Epoch {epoch+1}/{training_epochs},Loss:{avg_loss:.4f}')
            
    print(f'learning {model_name} was successfull')
    
    
def run_server(task_index):
    server = tf.train.Server(
        cluster_spec,
        job_name='worker',
        task_index =task_index
    )
    print(f'Worker {task_index} is ready...')
    server.join()
    


print('starting servers....')
        
server_threads = []

for i in range(3):
    t = threading.Thread(target=run_server,args=(i,))
    t.daemon = True
    t.start()
    server_threads.append(t)
    
time.sleep(3)



print('\n connecting to cluster....')

with tf.Session(target='grpc://localhost:2222',graph=graph) as sess:
    sess.run(init_op)
    print('Variables initialized')
    
    print('Starting parallel training for three models...\n')
    
    lock = threading.lock()
    
    thread_A = threading.Thread(
        target=train_model,
        args = (sess,train_op_A,loss_A,'Model A ', lock)
    )
    thread_B = threading.Thread(
        target=train_model,
        args=(sess,train_op_C,loss_C,'Model C',lock)
    )
    
    thread_C = threading.Thread(
        target=train_model,
        args=(sess,train_op_C,loss_C,'Model C',lock)
    )
    
    thread_A.start()
    thread_B.start()
    thread_C.start()
    
    
    thread_A.join()
    thread_B.join()
    thread_C.join()

    print("\n" + "="*60)
    print("Training of all models completed.")
    print("="*60 + "\n")
    
    
    print('Evaluation of individual models')
    
    test_x = mnist.test.images
    test_y = mnist.test.labels
    
    
    pred_A = sess.run(predictions_A,feed_dict={x:test_x})
    correct_A = np.equal(np.argmax(pred_A,1),np.argmax(test_y,1))
    accuracy_A = np.mean(correct_A.astype(float))

    print(f'accuracy model A : {accuracy_A*100:.2f}%')
    

    pred_B = sess.run(predictions_B, feed_dict={x: test_x})
    correct_B =   np.equal(np.argmax(pred_B,1),np.argmax(test_y,1))
    accuracy_B = np.mean(correct_B.astype(float))
    print(f"accuracy Model B: {accuracy_B*100:.2f}%")
    


    pred_C = sess.run(predictions_C, feed_dict={x: test_x})
    correct_C = np.equal(np.argmax(pred_C, 1), np.argmax(test_y, 1))
    accuracy_C = np.mean(correct_C.astype(float))
    print(f"accuracy Model C: {accuracy_C*100:.2f}%")
    

    print("\n" + "="*60)
    print("Evaluation Ensemble:")
    print("="*60)
    
    
    
    ensemble_acc =  sess.run(
        ensemble_accuracy,
        feed_dict = {x:test_x,y:test_y}
    )
    print(f"\a accuracy Ensemble : {ensemble_acc*100:.2f}%")
    
    print("\n" + "="*60)
    print("Examples of predictions:")
    print("="*60)
    
    
    sample_indices = np.random.choice(len(test_x),5,replace=False)
    
    
    for idx in sample_indices:
        sample_x = test_x[idx:idx+1]
        sample_y = test_y[idx:idx+1]
        
        pA, pB, pC = sess.run(
            [predictions_A, predictions_B, predictions_C],
            feed_dict={x: sample_x}
        )
        ensemble_pred = (pA + pB + pC) / 3.0
        
        
        
        true_label = np.argmax(sample_y[0])
        pred_A_label = np.argmax(pA[0])
        pred_B_label = np.argmax(pB[0])
        pred_C_label = np.argmax(pC[0])
        ensemble_label = np.argmax(ensemble_pred[0])
        
        print(f"\n samples {idx}:")
        print(f"  Actual label: {true_label}")
        print(f"   prediction Model A: {pred_A_label}")
        print(f"  prediction Model B: {pred_B_label}")
        print(f"  prediction Model C: {pred_C_label}")
        print(f"  prediction Ensemble: {ensemble_label}")
        print(f"  {'✓ True' if ensemble_label == true_label else '✗ False'}")

print("\n" + "="*60)
print("ended")
print("="*60)


