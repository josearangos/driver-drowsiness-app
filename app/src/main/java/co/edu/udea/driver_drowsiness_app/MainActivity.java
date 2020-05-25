package co.edu.udea.driver_drowsiness_app;

import androidx.appcompat.app.AppCompatActivity;

import android.content.Context;
import android.content.res.AssetFileDescriptor;
import android.media.MediaPlayer;
import android.os.Bundle;
import android.view.SurfaceView;
import android.view.WindowManager;
import android.widget.TextView;
import android.widget.Toast;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.JavaCameraView;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfRect;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;
import org.opencv.objdetect.Objdetect;
import org.tensorflow.lite.Interpreter;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.channels.FileChannel;

public class MainActivity extends AppCompatActivity implements CameraBridgeViewBase.CvCameraViewListener2 {
    File cascFile_face, cascFile_Reyes, cascFile_Leyes;
    CameraBridgeViewBase cameraBridgeViewBase;
    CascadeClassifier faceDetector, ReyesDetector, LeyesDetector;
    private  Mat mRgba, mGray;
    JavaCameraView javaCameraView;
    AssetFileDescriptor fileDescriptor;
    FileInputStream inputStream;
    FileChannel fileChannel;
    Interpreter interpreter;
    TextView score;
    int cont= 0;
    MediaPlayer alarm;


    Mat mRgbaF;
    Mat mRgbaT;



    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        score = (TextView)findViewById(R.id.score);



         alarm = MediaPlayer.create(this,R.raw.alarma);

        try {
            fileDescriptor = getAssets().openFd("cnnCat2.tflite");
            inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
            fileChannel = inputStream.getChannel();
            long startOffset = fileDescriptor.getStartOffset();
            long declaredLength = fileDescriptor.getDeclaredLength();

            ByteBuffer tfLiteFile = fileChannel.map(FileChannel.MapMode.READ_ONLY,startOffset,declaredLength);
            interpreter  = new Interpreter(tfLiteFile);
        } catch (IOException e) {
            e.printStackTrace();
        }

       /*
       *  cameraBridgeViewBase =(JavaCameraView) findViewById(R.id.cameraView);
        cameraBridgeViewBase.setCameraIndex(1);
        cameraBridgeViewBase.setVisibility(SurfaceView.VISIBLE);
        cameraBridgeViewBase.setCvCameraViewListener(this);
       * */


        javaCameraView =(JavaCameraView) findViewById(R.id.cameraView);
        javaCameraView.setCameraIndex(1);
        javaCameraView.setVisibility(SurfaceView.VISIBLE);




        if(!OpenCVLoader.initDebug()){
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_3_0_0, this,baseCallback);

        }else{
            try {
                baseCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
            } catch (IOException e) {
                e.printStackTrace();
            }
        }

        javaCameraView.setCvCameraViewListener(this);




    }

    @Override
    public void onCameraViewStarted(int width, int height) {
        //mRgba = new Mat();
        mGray = new Mat();

        mRgba = new Mat();
        mRgbaF = new Mat(height, width, CvType.CV_8UC4);
        mRgbaT = new Mat(width, height, CvType.CV_8UC4);


    }

    @Override
    public void onCameraViewStopped() {
        mRgba.release();
        mGray.release();
    }

    @Override
    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {
        mRgba = inputFrame.rgba();
        //mGray = inputFrame.gray();


        // Rotate mRgba 90 degrees
        Core.transpose(mRgba, mRgbaT);
        Imgproc.resize(mRgbaT, mRgbaF, mRgbaF.size(), 0,0, 0);
        Core.flip(mRgbaF, mRgba, 0 );
        Imgproc.cvtColor(mRgba,mGray,Imgproc.COLOR_RGB2GRAY);


        MatOfRect faceDetections = new MatOfRect();
        MatOfRect LeyesDetections = new MatOfRect();
        MatOfRect ReyesDetections = new MatOfRect();
        float P_Leyes = 0;
        float P_Reyes = 0;
        faceDetector.detectMultiScale(mGray,faceDetections);
        //ReyesDetector.detectMultiScale(mGray,ReyesDetections);
        for(Rect rect: faceDetections.toArray()){
            Imgproc.rectangle(mRgba,new Point(rect.x,rect.y),new Point(rect.x+rect.width, rect.y + rect.height),
                    new Scalar(0,255,0));
        }


        LeyesDetector.detectMultiScale(mGray,LeyesDetections);
        //Left Eyes
        for (Rect rect_leyes: LeyesDetections.toArray()){
            Imgproc.rectangle(mRgba,new Point(rect_leyes.x,rect_leyes.y),new Point(rect_leyes.x+rect_leyes.width,
                            rect_leyes.y + rect_leyes.height),
                    new Scalar(0,0,255));

            P_Leyes = predictModel(mRgba,rect_leyes);

        }
        //Rigth Eyes
        ReyesDetector.detectMultiScale(mGray,ReyesDetections);
        for(Rect rect_Reyes: ReyesDetections.toArray()){
               /*Imgproc.rectangle(face,new Point(rect_Reyes.x,rect_Reyes.y),new Point(rect_Reyes.x+rect_Reyes.width, rect_Reyes.y + rect_Reyes.height),
                    new Scalar(0,0,255));
                */
            P_Reyes = predictModel(mRgba,rect_Reyes);

        }
        if(P_Leyes == 1 &&   P_Reyes == 1){
            cont++;
        }else{
            cont--;
        }
        if(cont<0){
            cont=0;
        }
        score.setText("Score: "+String.valueOf(cont));

        if(cont >4){
            alarm.start();
        }

       return mRgba;
    }

    public float predictModel(Mat mRgba, Rect rect ){
        Mat leyes = mRgba.submat(rect);
        Imgproc.cvtColor(leyes, leyes, Imgproc.COLOR_RGB2GRAY);
        Size sz = new Size(24,24);
        Imgproc.resize(leyes,leyes, sz);
        float new_mat[][][][]  = new float[1][24][24][1];
        for (int i =0; i<leyes.size(0);i++) {
            for(int j = 0; j< leyes.size(1);j++){
                double[] aux = leyes.get(i,j);
                new_mat[0][i][j][0]=(float) aux[0];
            }
        }
        float[][] lpred =new float[1][2];

        interpreter.run(new_mat,lpred);

        return lpred[0][0];

    }


    @Override
    protected void onResume() {
        super.onResume();
        if (!OpenCVLoader.initDebug()){
            Toast.makeText(getApplicationContext(),"There's a problem, yo!", Toast.LENGTH_SHORT).show();
        }else{
            try {
                baseCallback.onManagerConnected(baseCallback.SUCCESS);
            } catch (IOException e) {
                e.printStackTrace();
            }

        }


    }




    @Override
    protected void onPause() {
        super.onPause();
        if(cameraBridgeViewBase!=null){
            cameraBridgeViewBase.disableView();
        }
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (cameraBridgeViewBase!=null){
            cameraBridgeViewBase.disableView();
        }
    }


    private BaseLoaderCallback  baseCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) throws IOException {
            super.onManagerConnected(status);

            switch (status){
                case LoaderCallbackInterface.SUCCESS:


                    //Face detection classifier
                    InputStream is_face = getResources().openRawResource(R.raw.haarcascade_frontalface_alt2);
                    File cascadeDir_faces = getDir("cascade", Context.MODE_PRIVATE);
                    cascFile_face  = new File(cascadeDir_faces,"haarcascade_frontalface_alt2.xml");
                    FileOutputStream fos_face = new FileOutputStream(cascFile_face);
                    byte[] buffer = new byte[4096];
                    int bytesRead;

                    while ((bytesRead = is_face.read(buffer)) != -1){
                        fos_face.write(buffer,0,bytesRead);
                    }
                    is_face.close();
                    fos_face.close();

                    // ------------------ load right eye classificator -----------------------

                    InputStream is_Reyes = getResources().openRawResource(R.raw.haarcascade_righteye_2splits);
                    File cascadeDir_Reyes = getDir("cascade", Context.MODE_PRIVATE);
                    cascFile_Reyes  = new File(cascadeDir_Reyes,"haarcascade_righteye_2splits.xml");
                    FileOutputStream fos_Reyes = new FileOutputStream(cascFile_Reyes);
                    byte[] bufferRE = new byte[4096];
                    int bytesReadER;
                    while ((bytesReadER = is_Reyes.read(bufferRE)) != -1){
                        fos_Reyes.write(bufferRE,0,bytesReadER);
                    }
                    is_Reyes.close();
                    fos_Reyes.close();

                    // ------------------ load left eye classificator -----------------------
                    InputStream is_Leyes = getResources().openRawResource(R.raw.haarcascade_lefteye_2splits);
                    File cascadeDir_Leyes = getDir("cascade", Context.MODE_PRIVATE);
                    cascFile_Leyes =  new File(cascadeDir_Leyes,"haarcascade_lefteye_2splits.xml");
                    FileOutputStream fos_Leyes = new FileOutputStream(cascFile_Leyes);

                    byte[] bufferEL = new byte[4096];

                    int bytesReadEL;
                    while ((bytesReadEL = is_Leyes.read(bufferEL)) != -1) {
                        fos_Leyes.write(bufferEL, 0, bytesReadEL);
                    }

                    is_Leyes.close();
                    fos_Leyes.close();


                    faceDetector = new CascadeClassifier(cascFile_face.getAbsolutePath());
                    ReyesDetector = new CascadeClassifier(cascFile_Reyes.getAbsolutePath());
                    LeyesDetector = new CascadeClassifier(cascFile_Leyes.getAbsolutePath());



                    if (faceDetector.empty()){
                        faceDetector = null;
                    }else{
                        //cascadeDir_faces.delete();
                    }

                    if (ReyesDetector.empty()){
                        ReyesDetector = null;
                    }else{
                        //cascadeDir_Reyes.delete();
                    }

                    if (LeyesDetector.empty()){
                        LeyesDetector = null;
                    }else{
                        //cascadeDir_Leyes.delete();
                    }


                    javaCameraView.enableFpsMeter();
                    javaCameraView.enableView();


                    break;
                default:
                    super.onManagerConnected(status);
                    break;

            }
        }


    };

}
