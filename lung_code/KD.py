#!/usr/bin/env python  
# -*- coding:utf-8 _*-
'''
@author:Chen Liuyin
@file: KD.py 
@time: 2022/01/12
@software: PyCharm 
'''
from implement_concat import generate_model, read_image
import os
import tensorflow as tf
from sklearn.model_selection import train_test_split
import numpy as np

# Average the loss across the batch size within an epoch
train_loss = tf.keras.metrics.Mean(name="train_loss")
valid_loss = tf.keras.metrics.Mean(name="test_loss")

# Specify the performance metric
train_acc = tf.keras.metrics.SparseCategoricalAccuracy(name="train_acc")
valid_acc = tf.keras.metrics.SparseCategoricalAccuracy(name="valid_acc")

def get_labeled_loss(student_logits, teacher_logits,
                true_labels, temperature=5.,
                alpha=0.5):
    teacher_probs = tf.nn.softmax(teacher_logits / temperature)
    kl_loss = tf.keras.losses.categorical_crossentropy(
        teacher_probs, student_logits / temperature,
        from_logits=True)

    ce_loss = tf.keras.losses.sparse_categorical_crossentropy(
        true_labels, student_logits, from_logits=True)

    total_loss = (alpha * kl_loss) + ((1-alpha) * ce_loss)
    return total_loss

def get_unlabeled_loss(student_logits, teacher_logits, temperature):
    teacher_probs = tf.nn.softmax(teacher_logits / temperature)
    kl_loss = tf.keras.losses.categorical_crossentropy(
        teacher_probs, student_logits / temperature,
        from_logits=True)
    return kl_loss


class labeled_KD(tf.keras.Model):
    def __init__(self, trained_teacher, student,
                 temperature=5., alpha=0.5):
        super(labeled_KD, self).__init__()
        self.trained_teacher = trained_teacher
        self.student = student
        self.temperature = temperature
        self.alpha = alpha

    def train_step(self, labeled_data):

        images, labels = labeled_data
        teacher_logits = self.trained_teacher(images)

        with tf.GradientTape() as tape:
            student_logits = self.student(images)
            loss = get_labeled_loss(student_logits, teacher_logits, labels, self.temperature, self.alpha)
        gradients = tape.gradient(loss, self.student.trainable_variables)
        # As mentioned in Section 2 of https://arxiv.org/abs/1503.02531
        gradients = [gradient * (self.temperature ** 2) for gradient in gradients]
        self.optimizer.apply_gradients(zip(gradients, self.student.trainable_variables))

        train_loss.update_state(loss)
        train_acc.update_state(labels, tf.nn.softmax(student_logits))
        t_loss, t_acc = train_loss.result(), train_acc.result()
        train_loss.reset_states(), train_acc.reset_states()
        return {"train_loss": t_loss, "train_accuracy": t_acc}

    def test_step(self, labeled_data):
        images, labels = labeled_data
        teacher_logits = self.trained_teacher(images)

        student_logits = self.student(images, training=False)
        loss = get_labeled_loss(student_logits, teacher_logits,
                           labels, self.temperature, self.alpha)

        valid_loss.update_state(loss)
        valid_acc.update_state(labels, tf.nn.softmax(student_logits))
        v_loss, v_acc = valid_loss.result(), valid_acc.result()
        valid_loss.reset_states(), valid_acc.reset_states()
        return {"loss": v_loss, "accuracy": v_acc}

class unlabeled_KD(tf.keras.Model):
    def __init__(self, trained_teacher, student,
                 temperature=5., alpha=0.5):
        super(unlabeled_KD, self).__init__()
        self.trained_teacher = trained_teacher
        self.student = student
        self.temperature = temperature
        self.alpha = alpha

    def train_step(self, unlabeled_data):

        images = unlabeled_data
        teacher_logits = self.trained_teacher(images)

        with tf.GradientTape() as tape:
            student_logits = self.student(images)
            loss = get_unlabeled_loss(student_logits, teacher_logits, self.temperature)
        gradients = tape.gradient(loss, self.student.trainable_variables)
        # As mentioned in Section 2 of https://arxiv.org/abs/1503.02531
        gradients = [gradient * (self.temperature ** 2) for gradient in gradients]
        self.optimizer.apply_gradients(zip(gradients, self.student.trainable_variables))

        train_loss.update_state(loss)
        train_acc.update_state(tf.nn.softmax(teacher_logits), tf.nn.softmax(student_logits))
        t_loss, t_acc = train_loss.result(), train_acc.result()
        train_loss.reset_states(), train_acc.reset_states()
        return {"train_loss": t_loss, "train_accuracy": t_acc}

    def test_step(self, unlabeled_data):
        images, labels = unlabeled_data
        teacher_logits = self.trained_teacher(images)

        student_logits = self.student(images, training=False)
        loss = get_unlabeled_loss(student_logits, teacher_logits, self.temperature)

        valid_loss.update_state(loss)
        valid_acc.update_state(tf.nn.softmax(teacher_logits), tf.nn.softmax(student_logits))
        v_loss, v_acc = valid_loss.result(), valid_acc.result()
        valid_loss.reset_states(), valid_acc.reset_states()
        return {"loss": v_loss, "accuracy": v_acc}

if __name__ == "__main__":

    x1,y1 = read_image(r"C:\Users\sdscit\Desktop\all\IAC","IAC")
    x2,y2 = read_image(r"C:\Users\sdscit\Desktop\all\MIA","MIA")
    x3,y3 = read_image(r"C:\Users\sdscit\Desktop\all\AIS","AIS")
    x1_train, x1_test, y1_train, y1_test = train_test_split(x1, y1, test_size=0.3, random_state=20)
    x2_train, x2_test, y2_train, y2_test = train_test_split(x2, y2, test_size=0.3, random_state=20)
    x3_train, x3_test, y3_train, y3_test = train_test_split(x3, y3, test_size=0.3, random_state=20)
    x_train = np.concatenate((x1_train,x2_train,x3_train))
    y_train  = np.concatenate((y1_train,y2_train,y3_train))
    x_test = np.concatenate((x1_test,x2_test,x3_test))
    y_test  = np.concatenate((y1_test,y2_test,y3_test))

    un_x = read_image(r"C:\Users\sdscit\Desktop\Dataset4", "")
    un_train, un_test = train_test_split(un_x, test_size=0.3, random_state=20)


    teacher_model = generate_model()
    teacher_model = teacher_model.load_model(os.path.join(r'C:\Users\sdscit\Desktop\ct_imgs\best_model'))

    model1 = labeled_KD(teacher_model, generate_model())
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
    model1.compile(optimizer)

    model1.fit([x_train, y_train],
                validation_data=[x_test, y_test],
                epochs=10)

    student = unlabeled_KD(model1, generate_model())
    student.compile(optimizer)
    student.fit([un_train], validation_data=[un_test], epochs=10)