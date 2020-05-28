package co.edu.udea.driver_drowsiness_app;

import androidx.appcompat.app.AppCompatActivity;
import java.util.concurrent.TimeUnit;

import android.Manifest;
import android.content.Context;
import android.content.pm.PackageManager;
import android.content.res.AssetFileDescriptor;
import android.media.MediaPlayer;
import android.os.Build;
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
    private final int REQUEST_CODE_ASK_PERMISSIONS = 123;
    int SLEEPINESS_THRESHOLD = 4;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        init_views();
        setting_model_files();
        request_permissions_camera();
    }

    private void init_opencv() {
        if(!OpenCVLoader.initDebug()){
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_3_0_0, this,baseCallback);

        }else{
            try {
                baseCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }


    public void init_views(){
        score = (TextView)findViewById(R.id.score);
        alarm = MediaPlayer.create(this,R.raw.alarma);
        javaCameraView =(JavaCameraView) findViewById(R.id.cameraView);
        javaCameraView.setCameraIndex(1);
        javaCameraView.setVisibility(SurfaceView.VISIBLE);
        javaCameraView.setCvCameraViewListener(this);

    }

    public void setting_model_files(){
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
    }

    public void request_permissions_camera(){
        int hasWriteContactsPermission = checkSelfPermission(Manifest.permission.CAMERA);
        if (hasWriteContactsPermission != PackageManager.PERMISSION_GRANTED) {
            // request permission
            Toast.makeText(this, "Soci@ dame permisos de la cámara", Toast.LENGTH_LONG).show();

            requestPermissions(new String[] {Manifest.permission.CAMERA},
                    REQUEST_CODE_ASK_PERMISSIONS);



        }else if (hasWriteContactsPermission == PackageManager.PERMISSION_GRANTED){
            init_opencv();
        }
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, String[] permissions, int[] grantResults) {
        if(REQUEST_CODE_ASK_PERMISSIONS == requestCode) {
            if (grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                init_opencv();
            } else {
                Toast.makeText(this, "Aja, y los permisos para cuando?", Toast.LENGTH_LONG).show();
            }
        }else{
            super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        }
    }

    @Override
    public void onCameraViewStarted(int width, int height) {
        mGray = new Mat(height,width,CvType.CV_8UC4);
        mRgba = new Mat(height,width,CvType.CV_8UC4);

    }

    @Override
    public void onCameraViewStopped() {
        mRgba.release();
        mGray.release();
    }

    @Override
    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {
        //Capturamos las imagenes en RGB y en escala de grises
        mRgba = inputFrame.rgba();
        mGray = inputFrame.gray();
        Imgproc.cvtColor(mRgba,mGray,Imgproc.COLOR_RGB2GRAY);

        //Definimos los objetos donde almacenamos los rectangulos de los objetos de interes(Rostro, Ojo izquierdo, Ojo derecho )
        MatOfRect faceDetections = new MatOfRect();
        MatOfRect LeyesDetections = new MatOfRect();
        MatOfRect ReyesDetections = new MatOfRect();
        //Definimos las variables para almacenar las prediciones del modelo
        float P_Leyes = 0;
        float P_Reyes = 0;
        //Usamos el objeto CascadeClassifier, para encontrar rostros
        faceDetector.detectMultiScale(mGray,faceDetections);
        //Dibujamos el rectangulo del rostro detectado
        for(Rect rect: faceDetections.toArray()){
        Imgproc.rectangle(mRgba,new Point(rect.x,rect.y),new Point(rect.x+rect.width, rect.y + rect.height),
                new Scalar(0,255,0));
        }
        //Usamos el objeto CascadeClassifier, para encontrar ojos izquierdos
        LeyesDetector.detectMultiScale(mGray,LeyesDetections);
        //Left Eyes
        //Dibujamos el rectangulo del ojo izquierdo detectado
        for (Rect rect_leyes: LeyesDetections.toArray()){
        Imgproc.rectangle(mRgba,new Point(rect_leyes.x,rect_leyes.y),new Point(rect_leyes.x+rect_leyes.width,
                        rect_leyes.y + rect_leyes.height),
                new Scalar(0,0,255));
            //Capturamos la predicion del modelo para el ojo izquierdo
            P_Leyes = predictModel(mRgba,rect_leyes);

        }
        //Usamos el objeto CascadeClassifier, para encontrar ojos derecho
        ReyesDetector.detectMultiScale(mGray,ReyesDetections);
        for(Rect rect_Reyes: ReyesDetections.toArray()){
            //Capturamos la predicion del modelo para el ojo derecho
            P_Reyes = predictModel(mRgba,rect_Reyes);

        }
        /*
        * Comparamos las prediciones de ambos ojos y si son iguales a 1 (Ojo Cerrado)
        * aumentamos en una unidad el score, si no disminuimos
        * */
        if(P_Leyes == 1 &&   P_Reyes == 1){
            cont++;
        }else{
            cont--;
        }
        if(cont<0){
            cont=0;
        }
        //Acutalizamos el score
        score.setText("Score: "+String.valueOf(cont));

        //Si el score es superior a un umbral de somnolencia (4), reproducimos el tono de alarma
        if(cont >SLEEPINESS_THRESHOLD){
            alarm.start();
        }

        return mRgba;
    }

    public float predictModel(Mat mRgba, Rect rect ){
        //Seleccionamos de la imagen complea la Region de interes
        Mat leyes = mRgba.submat(rect);
        //Convertimos a escala de grises la imagen
        Imgproc.cvtColor(leyes, leyes, Imgproc.COLOR_RGB2GRAY);
        /*
        * Debido a que el modelo que predice si un ojo esta abierto o cerrado,
        * solo recibe imagenes de dimension 24 x 24, debemos ajustar la imagen a esas
        * dimensiones
        * */
        Size sz = new Size(24,24);
        Imgproc.resize(leyes,leyes, sz);

        // Dada la forma del modelo, para poder predecir debemos enviar una tupla de (1,24,24,1)
        // por lo tanto ajustamos la imagen para que coincidan con dichas imagenes.
        float new_mat[][][][]  = new float[1][24][24][1];
        for (int i =0; i<leyes.size(0);i++) {
            for(int j = 0; j< leyes.size(1);j++){
                double[] aux = leyes.get(i,j);
                new_mat[0][i][j][0]=(float) aux[0];
            }
        }
        //Definimos el vector donde guardaremos la respuesta de las prediciones
        float[][] lpred =new float[1][2];

        //Ejecutamos y retornamos las prediciones
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


                   javaCameraView.enableView();


                    break;
                default:
                    super.onManagerConnected(status);
                    break;

            }
        }


    };

}
