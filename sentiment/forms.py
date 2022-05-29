from django import forms
from .models import *
  
class AnalysisForm(forms.ModelForm):
  
    class Meta:
        model = Analysis
        fields = ['name', 'userImage']