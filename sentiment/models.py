from django.db import models
from django.contrib.auth.models import User
# Create your models here.

class Analysis(models.Model):
    name = models.CharField(max_length=50)
    userImage = models.ImageField(upload_to="image/")

    def delete(self, using=None, keep_parents=False):
        self.name.storage.delete(self.name)
        self.userImage.storage.delete(self.userImage)
        super().delete()
