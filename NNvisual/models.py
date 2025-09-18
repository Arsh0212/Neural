from django.db import models

# Create your models here.
class NeuralNetwork(models.Model):

    ACTIVATION_CHOICES = [
        ('relu', 'ReLU'),
        ('sigmoid', 'Sigmoid'),
        ('tanh', 'Tanh'),
        ('linear', 'Linear'),
    ]

    epoch = models.IntegerField()
    batch_size = models.IntegerField()
    learning_rate = models.FloatField()
    activation_function = models.CharField(max_length=10, default='tanh' , choices=ACTIVATION_CHOICES)
    dataset = models.IntegerField()


    def __str__(self):
        return str(self.epoch)