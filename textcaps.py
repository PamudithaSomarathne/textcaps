import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from model.datapipe import getDataset
from model.capsule import TextCaps
import argparse
import json
import time

parser = argparse.ArgumentParser(description='TextCapsV2')
parser.add_argument('config', type=str, help='Configuration file path')
parser.add_argument('-v', action='store_true', default=False, help='Enable verbose')
parser.add_argument('-debug', action='store_true', default=False, help='Enable debug')
args = parser.parse_args()

with open(args.config) as f:
    config = json.load(f)

model_name, revision = config['model_name'], config['revision']

def train_step(x_train, y_train, rcn_weight):
    with tf.GradientTape() as tape:
        caps_class, rcn_out = model(x_train, y_train)
        cls_loss = cls_loss_fn(caps_class, y_train)
        rcn_loss = rcn_loss_fn(rcn_out, x_train)
        loss = cls_loss + rcn_weight*rcn_loss
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    cls_acc.update_state(tf.argmax(y_train, axis=1), tf.argmax(caps_class, axis=1))
    rcn_acc.update_state(x_train, rcn_out)
    return loss, cls_loss, rcn_loss

def train_model(train_set, val_set, epochs, val_epochs, record_logs, logs_path, load_model, save_weights, weights_path, rcn_weight):

    if not(args.v) & record_logs:
        train_log_dir = logs_path + model_name + '.' + str(revision) + '/'
        train_summary_writer = tf.summary.create_file_writer(train_log_dir)
        with train_summary_writer.as_default():
            tf.summary.text("Experiment details:", config["comment"], step=0)
            train_summary_writer.flush()
    
    if load_model:
        model.load_weights(weights_path + model_name + '.' + str(revision) + '/')
        print("Loaded from checkpoint")
    
    for epoch in range(1, epochs+1):
        for i,d in enumerate(train_set):
            x_train, y_train = d
            loss, cls_loss, rcn_loss = train_step(x_train, y_train, rcn_weight)
            if args.debug: break
    
            if args.v:
                print('step: {0} \t | total_loss: {1:.3f} \t | cls_loss: {2:.3f} \t | cls_accuracy: {3:.2f}% \t | rcn_loss: {4:.3f} \t | rcn_acc: {5:.2f}%'.format(
                    (epoch-1)*buffer_size//batch_size + i,
                    loss.numpy().item(),
                    cls_loss.numpy().item(),
                    cls_acc.result().numpy().item(),
                    rcn_loss.numpy().item(),
                    rcn_acc.result().numpy().item()
                ))
            
            if not(args.v) & record_logs:
                step = (epoch-1)*buffer_size//batch_size + i,
                with train_summary_writer.as_default():
                    tf.summary.scalar('Total loss', tf.squeeze(loss), step=step)
                    tf.summary.scalar('Classification loss', tf.squeeze(cls_loss), step=step)
                    tf.summary.scalar('Classification accuracy', cls_acc.result().numpy(), step=step)
                    tf.summary.scalar('Reconstruction loss', tf.squeeze(rcn_loss), step=step)
                    tf.summary.scalar('Reconstruction accuracy', rcn_acc.result().numpy(), step=step)
                    train_summary_writer.flush()
    
        if save_weights:
            model.save_weights(weights_path + model_name + '.' + str(revision) + '/')
        
        cls_acc.reset_state()
        rcn_acc.reset_state()

        if val_epochs>0 and epoch%val_epochs==0:
            for d in val_set:
                x_test, y_test = d
                caps_class, rcn_out = model(x_test)
                cls_loss = cls_loss_fn(caps_class, y_test)
                rcn_loss = rcn_loss_fn(rcn_out, x_test)
                loss = cls_loss + rcn_loss
                cls_acc.update_state(tf.argmax(y_train, axis=1), tf.argmax(caps_class, axis=1))
                rcn_acc.update_state(x_train, rcn_out)
            
            if not(args.v) & record_logs:
                with train_summary_writer.as_default():
                    tf.summary.scalar('Test classification accuracy', cls_acc.result().numpy(), step=epoch)
                    tf.summary.scalar('Test reconstruction accuracy', rcn_acc.result().numpy(), step=epoch)
                    train_summary_writer.flush()
            
            print('Test classification accuracy: {0:.2f}% \t | Test reconstruction accuracy: {1:.2f}%'.format(
                cls_acc.result().numpy(),
                rcn_acc.result().numpy()
            ))

            cls_acc.reset_state()
            rcn_acc.reset_state()

def margin_loss(y_pred, y_true):
    loss = tf.reduce_mean(tf.reduce_sum(tf.multiply(y_true, tf.square(tf.maximum(0.0, 0.9-y_pred))) + \
                                        0.5 * tf.multiply((1-y_true), tf.square(tf.maximum(0.0, y_pred-0.1))), axis=-1, keepdims=True), keepdims=True)
    return loss

if __name__ == '__main__':

    model = TextCaps(**config['model'])

    cls_loss_fn = margin_loss
    rcn_loss_fn = tf.keras.losses.MeanSquaredError()

    cls_acc = tf.keras.metrics.Accuracy()
    rcn_acc = tf.keras.metrics.Accuracy()

    optimizer = tf.keras.optimizers.Adam(learning_rate=config['learning_rate'])

    x_train, y_train, x_test, y_test = getDataset(**config['dataset'])
    batch_size = config['datapipe']['batch_size']
    buffer_size = len(x_train)
    t_set = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    t_set = t_set.shuffle(buffer_size, reshuffle_each_iteration=True).batch(batch_size)

    v_set = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size)

    print(model.summary())

    print("Starting training for {} epochs...".format(config['train']['epochs']))
    t1 = time.time()
    train_model(t_set, v_set, **config['train'])
    t2 = time.time()
    print("Training completed in {} seconds".format(t2 - t1))
