from django.shortcuts import render

# Create your views here.

def home(request):
    return render(request, 'main/home.html')

def algorithm1(request):
    return render(request, 'main/algorithm1.html')

def algorithm2(request):
    return render(request, 'main/algorithm2.html')

def algorithm3(request):
    return render(request, 'main/algorithm3.html')

def algorithm4(request):
    return render(request, 'main/algorithm4.html')

def algorithm5(request):
    return render(request, 'main/algorithm5.html')

def about(request):
    return render(request, 'main/about.html')

def services(request):
    return render(request, 'main/services.html')

def contact(request):
    return render(request, 'main/contact.html')

def blog(request):
    return render(request, 'main/blog.html')

def dashboard(request):
    return render(request, 'main/dashboard.html')
