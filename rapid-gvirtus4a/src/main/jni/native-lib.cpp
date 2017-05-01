#include <jni.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>

extern "C"
JNIEXPORT jstring JNICALL
Java_eu_project_rapid_gvirtus4a_Buffer_prepareFloat(JNIEnv *env, jobject instance,
                                                  jfloatArray floats_) {
    jstring result=NULL;
    size_t i;
    size_t j;
    char *msg=NULL;
    int msg_index=0;
    char buffer[9];
    unsigned char const *p;
    jsize len = env->GetArrayLength(floats_);
    long msg_size=4*len*sizeof(float)+1;
    msg=(char *)malloc(msg_size);
    msg[msg_size-1]=0x00;

    int ii;
    union {
        char a;
        unsigned char bytes[4];
    } thing;

    thing.a = '0';

    jfloat *body = env->GetFloatArrayElements(floats_, 0);
    for (j=0;j<len;j++) {
        p = (unsigned char const *)&body[j];
        for (i = 0; i != sizeof(float); ++i) {
            sprintf(buffer,"%02X%02X%02X%02X",p[0],p[1],p[2],p[3]);
        }
        memcpy(msg+8*j,buffer,8);
    }
    env->ReleaseFloatArrayElements(floats_, body, 0);
    result = env->NewStringUTF(msg);
    free(msg);
    return result;
}

extern "C"
JNIEXPORT jstring JNICALL
Java_eu_project_rapid_gvirtus4a_Buffer_preparePtxSource(JNIEnv *env, jobject instance,
                                                      jstring ptxSource_, jlong size) {
    size_t i,j;
    char *msg=NULL;
    long msg_size=size*2+1;
    msg=(char *)malloc(msg_size);
    msg[msg_size-1]=0x00;


    jstring result=NULL;

    const char *nativeString = env->GetStringUTFChars(ptxSource_, 0);

    for(i = 0; i<size; i++){
        sprintf(msg+i*2, "%02X", nativeString[i]);
    }


//printf("name printed as %%s is %s\n",msg);
/*printf("name printed as %%s is %s\n",nativeString);
printf("name printed as %%c 0 is %c\n",nativeString[0]);
printf("name printed as %%c 1 is %c\n",nativeString[1]);
printf("name printed as %%c 2 is %c\n",nativeString[2]);
printf("name printed as %%c 3 is %c\n",nativeString[3]);
printf("name printed as %%x 0 is %02X\n",nativeString[0] & 0xff);
printf("name printed as %%x 1 is %02X\n",nativeString[1] & 0xff);
printf("name printed as %%x 2is %02X\n",nativeString[2] & 0xff);
printf("name printed as %%x 3 is %02X\n",nativeString[3] & 0xff);*/

    /*  int i, len;

      printf("Intro word:");
      fgets(word, sizeof(word), stdin);
      len = strlen(word);
      if(word[len-1]=='\n')
          word[--len] = '\0';*/



    env->ReleaseStringUTFChars(ptxSource_, nativeString);
    result = env->NewStringUTF(msg);
    free(msg);

    return result;

}

extern "C"
JNIEXPORT jstring JNICALL
Java_eu_project_rapid_gvirtus4a_Buffer_prepareSingleByte(JNIEnv *env, jobject instance, jint i) {

    char *msg=NULL;
    long msg_size=2+1;
    msg=(char *)malloc(msg_size);
    msg[msg_size-1]=0x00;
    jstring result=NULL;
    sprintf(msg, "%02X", i);
    result = env->NewStringUTF(msg);
    free(msg);

    return result;
}

