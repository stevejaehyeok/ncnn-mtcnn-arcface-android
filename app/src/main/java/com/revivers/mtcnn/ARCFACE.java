package com.revivers.mtcnn;

public class ARCFACE {

    public native boolean FeatureExtractionModelInit(String ModelPath);

    public native float[] getFeature(byte[] imageData_, int imageWidth, int imageHeight, int imageChannel, int[] faceInfo);

    static {
        System.loadLibrary("combined_jni");
    }
}
