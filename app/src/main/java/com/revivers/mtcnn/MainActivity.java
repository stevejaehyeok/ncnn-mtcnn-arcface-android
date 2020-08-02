// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2017 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

package com.revivers.mtcnn;

import android.Manifest;
import android.app.Activity;
import android.support.annotation.NonNull;
import android.support.v4.app.ActivityCompat;
import android.support.v4.content.ContextCompat;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.net.Uri;
import android.os.Bundle;
import android.os.Environment;
import android.text.TextUtils;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;

public class MainActivity extends Activity
{
    private static final int SELECT_IMAGE = 1;

    private TextView infoResult;
    private ImageView imageView;
    private Bitmap yourSelectedImage = null;

    //AppCompatEditText etMinFaceSize,etTestTimeCount,etThreadsNumber;
    private int minFaceSize = 40;
    private int testTimeCount = 10;
    private int threadsNumber = 4;

    private boolean maxFaceSetting = false;

    private MTCNN mtcnn = new MTCNN();

    private ARCFACE arcface = new ARCFACE();

    private int[] currentFaceInfo = new int[14];

    private float[] prevFeature = null;
    private boolean firstTime = true;

    //private GoogleApiClient client;

    private static final int REQUEST_EXTERNAL_STORAGE = 1;
    private static String[] PERMISSIONS_STORAGE = {
            "android.permission.READ_EXTERNAL_STORAGE",
            "android.permission.WRITE_EXTERNAL_STORAGE" };

    /** Called when the activity is first created. */
    @Override
    public void onCreate(Bundle savedInstanceState)
    {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.main);

        checkSelfPermission();
        //verifyStoragePermissions(this);
        //拷贝模型到sk卡
        try {
            copyBigDataToSD("det1.bin");
            copyBigDataToSD("det2.bin");
            copyBigDataToSD("det3.bin");
            copyBigDataToSD("det1.param");
            copyBigDataToSD("det2.param");
            copyBigDataToSD("det3.param");
            copyBigDataToSD("mobilefacenet.bin");
            copyBigDataToSD("mobilefacenet.param");
            Log.i("Temp tag", "Succeeded to load the weights");
        } catch (IOException e) {
            e.printStackTrace();
            Log.i("Temp tag", "Failed to load the weights");
        }
        //模型初始化

        File sdDir = Environment.getExternalStorageDirectory();//获取跟目录
        String sdPath = sdDir.toString() + "/mtcnn/";
//        String sdPath = this.getAssets().toString();
//        String sdPath = "src/main/assets/";
        Log.i("sdPath",sdPath);
        mtcnn.FaceDetectionModelInit(sdPath);


        // Arcface model initialization
        if (arcface.FeatureExtractionModelInit(sdPath)) {
            Log.i("Temp tag", "ArcFace model successfully initialized");
        }
        else {
            Log.i("Temp tag", "FAILED TO INITIALIZE THE ARCFACE MODEL!!!!!!!!");
        }

        //Log.i("isMTCNN", isMTCNN + " ");

        infoResult = (TextView) findViewById(R.id.infoResult);
        imageView = (ImageView) findViewById(R.id.imageView);

        Button buttonImage = (Button) findViewById(R.id.buttonImage);
        buttonImage.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View arg0) {
                Intent i = new Intent(Intent.ACTION_PICK);
                i.setType("image/*");
                startActivityForResult(i, SELECT_IMAGE);
            }
        });

        Button buttonDetect = (Button) findViewById(R.id.buttonDetect);
        buttonDetect.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View arg0) {
                if (yourSelectedImage == null)
                    return;

                minFaceSize = Integer.valueOf("40");
                testTimeCount = Integer.valueOf("10");
                threadsNumber = Integer.valueOf("4");

                if (threadsNumber != 1&&threadsNumber != 2&&threadsNumber != 4&&threadsNumber != 8){
                    //Log.i(TAG, "线程数："+threadsNumber);
                    infoResult.setText("线程数必须是（1，2，4，8）之一");
                    return;
                }

                //Log.i(TAG, "最小人脸："+minFaceSize);
                mtcnn.SetMinFaceSize(minFaceSize);
                mtcnn.SetTimeCount(testTimeCount);
                mtcnn.SetThreadsNumber(threadsNumber);

                //检测流程
                int width = yourSelectedImage.getWidth();
                int height = yourSelectedImage.getHeight();
                byte[] imageDate = getPixelsRGBA(yourSelectedImage);

                long timeDetectFace = System.currentTimeMillis();
                int faceInfo[] = null;
                if(!maxFaceSetting) {
                    faceInfo = mtcnn.FaceDetect(imageDate, width, height, 4);
                    //Log.i(TAG, "检测所有人脸");
                }
                else{
                    faceInfo = mtcnn.MaxFaceDetect(imageDate, width, height, 4);
                    //Log.i(TAG, "检测最大人脸");
                    for(int i = 0; i < 14; i++) {
                        currentFaceInfo[i] = faceInfo[i+1];
                    }
                }
                timeDetectFace = System.currentTimeMillis() - timeDetectFace;
                //Log.i(TAG, "人脸平均检测时间："+timeDetectFace/testTimeCount);

                if(faceInfo.length>1){
                    int faceNum = faceInfo[0];
                    infoResult.setText("图宽："+width+"高："+height+"人脸平均检测时间："+timeDetectFace/testTimeCount+" 数目：" + faceNum);
                    //Log.i(TAG, "图宽："+width+"高："+height+" 人脸数目：" + faceNum );

                    Bitmap drawBitmap = yourSelectedImage.copy(Bitmap.Config.ARGB_8888, true);
                    for(int i=0;i<faceNum;i++) {
                        int left, top, right, bottom;
                        Canvas canvas = new Canvas(drawBitmap);
                        Paint paint = new Paint();
                        left = faceInfo[1+14*i];
                        top = faceInfo[2+14*i];
                        right = faceInfo[3+14*i];
                        bottom = faceInfo[4+14*i];
                        paint.setColor(Color.RED);
                        paint.setStyle(Paint.Style.STROKE);//不填充
                        paint.setStrokeWidth(5);  //线的宽度
                        canvas.drawRect(left, top, right, bottom, paint);
                        //画特征点
                        canvas.drawPoints(new float[]{faceInfo[5+14*i],faceInfo[10+14*i],
                                faceInfo[6+14*i],faceInfo[11+14*i],
                                faceInfo[7+14*i],faceInfo[12+14*i],
                                faceInfo[8+14*i],faceInfo[13+14*i],
                                faceInfo[9+14*i],faceInfo[14+14*i]}, paint);//画多个点
                    }
                    imageView.setImageBitmap(drawBitmap);
                }else{
                    infoResult.setText("No Face Detected");
                }
            }
        });

        Button buttonDetectGPU = (Button) findViewById(R.id.buttonExtract);
        buttonDetectGPU.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View arg0) {

                if(yourSelectedImage == null) {
                    return;
                }

                int width = yourSelectedImage.getWidth();
                int height = yourSelectedImage.getHeight();
                byte[] imageData = getPixelsRGBA(yourSelectedImage);

                float[] feature = arcface.getFeature(imageData, width, height, 4, currentFaceInfo);

                if(firstTime) {
                    infoResult.setText("No previous feature to compare");
                    firstTime = false;
                }
                else {
                    double sim = 0.0;
                    for (int i = 0; i < feature.length; i++)
                        sim += feature[i] * prevFeature[i];
                    String result = Double.toString(sim);
                    infoResult.setText(result);
                }

                prevFeature = feature;
            }
        });

        Button testButton = (Button) findViewById(R.id.testButton);
        testButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View arg0) {
                /*String result = squeezencnn.Add();
                infoResult.setText(result);*/
            }
        });
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        //권한을 허용 했을 경우
        if(requestCode == 1){
            int length = permissions.length;
            for (int i = 0; i < length; i++) {
                if (grantResults[i] == PackageManager.PERMISSION_GRANTED) {
                    // 동의
                    Log.d("MainActivity","권한 허용 : " + permissions[i]);
                }
            }
        }
    }

    public void checkSelfPermission() {
        String temp = ""; //파일 읽기 권한 확인
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.READ_EXTERNAL_STORAGE) != PackageManager.PERMISSION_GRANTED) {
            temp += Manifest.permission.READ_EXTERNAL_STORAGE + " "; } //파일 쓰기 권한 확인

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.WRITE_EXTERNAL_STORAGE) != PackageManager.PERMISSION_GRANTED) {
            temp += Manifest.permission.WRITE_EXTERNAL_STORAGE + " "; } if (TextUtils.isEmpty(temp) == false) {
            // 권한 요청
            ActivityCompat.requestPermissions(this, temp.trim().split(" "),1); }
        else { // 모두 허용 상태
            Toast.makeText(this, "권한을 모두 허용", Toast.LENGTH_SHORT).show();
        }
    }


    public static void verifyStoragePermissions(Activity activity) {

        try {
            //检测是否有写的权限
            int permission = ActivityCompat.checkSelfPermission(activity,
                    "android.permission.WRITE_EXTERNAL_STORAGE");
            if (permission != PackageManager.PERMISSION_GRANTED) {
                // 没有写的权限，去申请写的权限，会弹出对话框
                ActivityCompat.requestPermissions(activity, PERMISSIONS_STORAGE,REQUEST_EXTERNAL_STORAGE);
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }


    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data)
    {
        super.onActivityResult(requestCode, resultCode, data);

        if (resultCode == RESULT_OK && null != data) {
            Uri selectedImage = data.getData();

            try
            {
                if (requestCode == SELECT_IMAGE) {
                    Bitmap bitmap = decodeUri(selectedImage);

                    Bitmap rgba = bitmap.copy(Bitmap.Config.ARGB_8888, true);

                    // resize to 227x227
                    //yourSelectedImage = Bitmap.createScaledBitmap(rgba, 227, 227, false);
                    yourSelectedImage = Bitmap.createScaledBitmap(rgba, rgba.getHeight(), rgba.getWidth(), false);

                    rgba.recycle();

                    imageView.setImageBitmap(bitmap);
                }
            }
            catch (FileNotFoundException e)
            {
                Log.e("MainActivity", "FileNotFoundException");
                return;
            }
        }
    }

    private Bitmap decodeUri(Uri selectedImage) throws FileNotFoundException
    {
        // Decode image size
        BitmapFactory.Options o = new BitmapFactory.Options();
        o.inJustDecodeBounds = true;
        BitmapFactory.decodeStream(getContentResolver().openInputStream(selectedImage), null, o);

        // The new size we want to scale to
        final int REQUIRED_SIZE = 400;

        // Find the correct scale value. It should be the power of 2.
        int width_tmp = o.outWidth, height_tmp = o.outHeight;
        int scale = 1;
        while (true) {
            if (width_tmp / 2 < REQUIRED_SIZE
               || height_tmp / 2 < REQUIRED_SIZE) {
                break;
            }
            width_tmp /= 2;
            height_tmp /= 2;
            scale *= 2;
        }

        // Decode with inSampleSize
        BitmapFactory.Options o2 = new BitmapFactory.Options();
        o2.inSampleSize = scale;
        return BitmapFactory.decodeStream(getContentResolver().openInputStream(selectedImage), null, o2);
    }

    //提取像素点
    private byte[] getPixelsRGBA(Bitmap image) {
        // calculate how many bytes our image consists of
        int bytes = image.getByteCount();
        ByteBuffer buffer = ByteBuffer.allocate(bytes); // Create a new buffer
        image.copyPixelsToBuffer(buffer); // Move the byte data to the buffer
        byte[] temp = buffer.array(); // Get the underlying array containing the

        return temp;
    }

    private void copyBigDataToSD(String strOutFileName) throws IOException {
        //Log.i(TAG, "start copy file " + strOutFileName);
        File sdDir = Environment.getExternalStorageDirectory();//获取跟目录
        File file = new File(sdDir.toString()+"/mtcnn/");
        if (!file.exists()) {
            file.mkdir();
        }

        String tmpFile = sdDir.toString()+"/mtcnn/" + strOutFileName;
        File f = new File(tmpFile);
        if (f.exists()) {
            //Log.i(TAG, "file exists " + strOutFileName);
            return;
        }
        InputStream myInput;
        java.io.OutputStream myOutput = new FileOutputStream(sdDir.toString()+"/mtcnn/"+ strOutFileName);
        myInput = this.getAssets().open(strOutFileName);
        byte[] buffer = new byte[1024];
        int length = myInput.read(buffer);
        while (length > 0) {
            myOutput.write(buffer, 0, length);
            length = myInput.read(buffer);
        }
        myOutput.flush();
        myInput.close();
        myOutput.close();
        //Log.i(TAG, "end copy file " + strOutFileName);

    }

}
