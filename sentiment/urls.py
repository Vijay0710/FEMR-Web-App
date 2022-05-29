from django.urls import path
from . import views
from django.conf import settings
from django.conf.urls.static import static
urlpatterns = [
    path('hello/',views.say_hello),
    path('demo/',views.demo),
    path('success/',views.success, name = 'success'),
    path('demo/<int:pk>/',views.delete_image,name='delete_image')
]

if settings.DEBUG:
    urlpatterns+=static(settings.MEDIA_URL,document_root=settings.MEDIA_ROOT)

