# 장고에 대한 정리
 
## 기본정리
https://tutorial.djangogirls.org/ko/ 참고!

####- 기본
1. 아나콘다 설치
2. 파이참 설치
***
***
 ####- 시작하기

**1.가상환경 만들기**

 ```
 > mkdir django_py
 > cd django_py
 
 > python -m venv myenv
 ```
 
**2.가상환경 사용하기**

```
 > myenv\Scripts\activate
 // (myenv) c:..... > 이렇게 변경되면 가상환경 적용된 것. 
```

 - 패키지 목록 저장하기 & 설치하기
 

freeze 를 사용하면, 현재까지 설치한 패키지의 목록들을 저장 할 수 있습니다.
pip install ~~~ 해줄때마다 실행!!
 
```
(가상환경) pip freeze > fileList.txt
```

지금까지 설치 한 목록을 fileList.txt에 저장 하는 명령어 입니다.
fileList.txt가 있으면, 지금 설치되어있는 패키지의 리스트를 동일하게 설치 할 수 있습니다.

 
```
(가상환경) pip install -r fileList.txt
```
fileList.txt에 저장되어있는 목록을 "-r" 옵션을 사용해서 설치 했습니다.

**3.장고 설치**

```
관리자권한으로 cmd 실행.
 > python -m pip install --upgrade pip
 > python -m pip install django~=2.0.0
 // 2.0대 버전을 다운 받겠다는것. 현재 3.0 이상 나옴.
```
 
**4.프로젝트 만들기**

```
 > myenv\Scripts\django-admin.exe startproject mysite .
 // 가상환경 내에 장고 어드민 파일로 프로젝트 생성
 // 마지막에 나온 . 은 현재 폴더를 의미하는 것임
```
 
 프로젝트 구조
 ```
django_py
├───myenv  (가상환경 파일들)
└───mysite
        settings.py
        urls.py
        wsgi.py
        __init__.py
├───manage.py
```

**5.설정 변경**

```
mysite/settings.py

LANGUAGE_CODE='ko'
TIME_ZONE='Asia/Seoul'
STATIC_ROOT=os.path.join(BASE_DIR, 'static') // 새로 추가
//ALLOWED_HOSTS = ['127.0.0.1', '.pythonanywhere.com'] //호스트 추가가 필요하다면. 
```

디비는 기본으로 sqlite3로 설정되어있음. 추후 변경가능하겟지?
디비를 생성하고자하면 아래 명령어 실행
 
```
 > python manage.py migrate
```

그리고 서버를 실행하고자 하면
```
 > python manage.py runserver
 > python manage.py runserver 0.0.0.0:62000
 > python manage.py runserver 192.168.0.101:62111 // 세팅에서 호스트를 추가한다면..
```
이와 같이 실행할수 있음.

***
***
##프로젝트 시작


**6.블로그 만들기**


```
 > python manage.py startapp blog(만들고자 하는 웹 이름)
```
실행하면 아래같은 구조가 만들어짐
```
    django_py
    ├── mysite
    ├── manage.py
    └── blog
        ├── migrations
        ├── __init__.py
        ├── admin.py
        ├── app.py
        ├── models.py
        ├── tests.py
        └── views.py
```
이거를 또 세팅에 추가해줘야함
```
mysite/settings.py
INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'blog',               // 요거
]
```

**7.모델 제작.**

blog/model.py 파일로 이동해서 장고 모델을 제작.

프로퍼티와 메소드를 통해서 모델을 제작하기.

이후 모델을 디비에 저장하기
```
 > python manage.py makemigrations blog
 // 블로그 프로젝트의 모델을 만드는 명령어
 
Migrations for 'blog':
  blog/migrations/0001_initial.py:
  - Create model Post
이런 결과가 뜸.
```
이를 통해 blog/migrations 폴더 밑에 이니셜파일이 하나 생성됨.
이제 
```
 > python manage.py migrate blog
 
Operations to perform:
  Apply all migrations: blog
Running migrations:
  Applying blog.0001_initial... OK
```
명령어를 통해 바로 반영할수있는 마이그레이션 파일이 생성된것.

**8.관리자모드**

관리자에서 모델링한 글을 추가수정삭제 가능함.

blog/admin.py를 열어 해당 코드를 추가

```
admin.site.register(Post)
```
앞에서 정의한 Post를  admin site에 가져옴.

그리고 슈퍼유저를 생성해야 로그인이 가능해짐.

```
(myvenv) > python manage.py createsuperuser

Username: admin
Email address: admin@admin.com
Password:
Password (again):
Superuser created successfully.
// 해당 라인에 맞게 입력
```
완료되면 로그인 가능함.

**9.URL 설정**


mysite/urls.py 파일을 확인해보면 이미 어드민에 대한 내용이 설정되어있음.

여기에 새로운 코드를 추가해주기.
```
urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('blog.urls')), // blog 폴더밑에 urls 파일을 생성해주기.
]
```


