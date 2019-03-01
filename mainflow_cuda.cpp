#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/utility.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/cudaoptflow.hpp"
#include "opencv2/cudaarithm.hpp"

using namespace std;
using namespace cv;
using namespace cv::cuda;

inline bool isFlowCorrect(Point2f u)
{
    return !cvIsNaN(u.x) && !cvIsNaN(u.y) && fabs(u.x) < 1e9 && fabs(u.y) < 1e9;
}

static Vec3b computeColor(float fx, float fy)
{
    static bool first = true;

    // relative lengths of color transitions:
    // these are chosen based on perceptual similarity
    // (e.g. one can distinguish more shades between red and yellow
    //  than between yellow and green)
    const int RY = 15;
    const int YG = 6;
    const int GC = 4;
    const int CB = 11;
    const int BM = 13;
    const int MR = 6;
    const int NCOLS = RY + YG + GC + CB + BM + MR;
    static Vec3i colorWheel[NCOLS];

    if (first)
    {
        int k = 0;

        for (int i = 0; i < RY; ++i, ++k)
            colorWheel[k] = Vec3i(255, 255 * i / RY, 0);

        for (int i = 0; i < YG; ++i, ++k)
            colorWheel[k] = Vec3i(255 - 255 * i / YG, 255, 0);

        for (int i = 0; i < GC; ++i, ++k)
            colorWheel[k] = Vec3i(0, 255, 255 * i / GC);

        for (int i = 0; i < CB; ++i, ++k)
            colorWheel[k] = Vec3i(0, 255 - 255 * i / CB, 255);

        for (int i = 0; i < BM; ++i, ++k)
            colorWheel[k] = Vec3i(255 * i / BM, 0, 255);

        for (int i = 0; i < MR; ++i, ++k)
            colorWheel[k] = Vec3i(255, 0, 255 - 255 * i / MR);

        first = false;
    }

    const float rad = sqrt(fx * fx + fy * fy);
    const float a = atan2(-fy, -fx) / (float) CV_PI;

    const float fk = (a + 1.0f) / 2.0f * (NCOLS - 1);
    const int k0 = static_cast<int>(fk);
    const int k1 = (k0 + 1) % NCOLS;
    const float f = fk - k0;

    Vec3b pix;

    for (int b = 0; b < 3; b++)
    {
        const float col0 = colorWheel[k0][b] / 255.0f;
        const float col1 = colorWheel[k1][b] / 255.0f;

        float col = (1 - f) * col0 + f * col1;

        if (rad <= 1)
            col = 1 - rad * (1 - col); // increase saturation with radius
        else
            col *= .75; // out of range

        pix[2 - b] = static_cast<uchar>(255.0 * col);
    }

    return pix;
}

static void drawOpticalFlow(const Mat_<float>& flowx, const Mat_<float>& flowy, Mat& dst, float maxmotion = -1)
{
    dst.create(flowx.size(), CV_8UC3);
    dst.setTo(Scalar::all(0));

    // determine motion range:
    float maxrad = maxmotion;

    if (maxmotion <= 0)
    {
        maxrad = 1;
        for (int y = 0; y < flowx.rows; ++y)
        {
            for (int x = 0; x < flowx.cols; ++x)
            {
                Point2f u(flowx(y, x), flowy(y, x));

                if (!isFlowCorrect(u))
                    continue;

                maxrad = max(maxrad, sqrt(u.x * u.x + u.y * u.y));
            }
        }
    }

    for (int y = 0; y < flowx.rows; ++y)
    {
        for (int x = 0; x < flowx.cols; ++x)
        {
            Point2f u(flowx(y, x), flowy(y, x));

            if (isFlowCorrect(u))
                dst.at<Vec3b>(y, x) = computeColor(u.x / maxrad, u.y / maxrad);
        }
    }
}

void convertFlowToImage(const Mat &flow_x, const Mat &flow_y, Mat &img_x, Mat &img_y,
                               double lowerBound, double higherBound) {
#define CAST(v, L, H) ((v) > (H) ? 255 : (v) < (L) ? 0 : cvRound(255*((v) - (L))/((H)-(L))))
    for (int i = 0; i < flow_x.rows; ++i) {
        for (int j = 0; j < flow_y.cols; ++j) {
            float x = flow_x.at<float>(i,j);
            float y = flow_y.at<float>(i,j);
            img_x.at<uchar>(i,j) = CAST(x, lowerBound, higherBound);
            img_y.at<uchar>(i,j) = CAST(y, lowerBound, higherBound);
        }
    }
#undef CAST
}

static cv::Mat showFlow(const GpuMat& d_flow)
{
    GpuMat planes[2];
    cuda::split(d_flow, planes);

    Mat flowx(planes[0]);
    Mat flowy(planes[1]);

    Mat out;
    drawOpticalFlow(flowx, flowy, out, 10);

    return out;
}

void drawOptFlowMap(const Mat& flow, Mat& cflowmap, int step,double, const Scalar& color){
    for(int y = 0; y < cflowmap.rows; y += step)
        for(int x = 0; x < cflowmap.cols; x += step)
        {
            const Point2f& fxy = flow.at<Point2f>(y, x);
            line(cflowmap, Point(x,y), Point(cvRound(x+fxy.x), cvRound(y+fxy.y)),
                 color);
            circle(cflowmap, Point(x,y), 2, color, -1);
        }
}

void encodeFlowMap(const Mat& flow_map_x, const Mat& flow_map_y,
                   vector<uchar>& encoded_x, vector<uchar>& encoded_y,
                   int bound, bool to_jpg){
    Mat flow_img_x(flow_map_x.size(), CV_8UC1);
    Mat flow_img_y(flow_map_y.size(), CV_8UC1);

    convertFlowToImage(flow_map_x, flow_map_y, flow_img_x, flow_img_y,
                       -bound, bound);

    if (to_jpg) {
        imencode(".jpg", flow_img_x, encoded_x);
        imencode(".jpg", flow_img_y, encoded_y);
    }else {
        encoded_x.resize(flow_img_x.total());
        encoded_y.resize(flow_img_y.total());
        memcpy(encoded_x.data(), flow_img_x.data, flow_img_x.total());
        memcpy(encoded_y.data(), flow_img_y.data, flow_img_y.total());
    }
}

Mat createSingleImg(vector<uchar>& encoded_x, vector<uchar>& encoded_y, cv::Size img_size){
    Mat channelR(img_size, CV_8UC1, cv::Scalar(0));
    Mat channelG(img_size, CV_8UC1, reinterpret_cast<char*>(encoded_y.data()));
    Mat channelB(img_size, CV_8UC1, reinterpret_cast<char*>(encoded_x.data()));

    // Invert channels,
    // don't copy data, just the matrix headers
    std::vector<Mat> channels;
    channels.push_back(channelB);
    channels.push_back(channelG);
    channels.push_back(channelR);

    // Create the output matrix
    Mat outputMat;
    merge(channels, outputMat);

    return outputMat;
}



int main(int argc, char** argv)
{
    // Need at least one argument
    if (argc < 3)
    {
        cout << "Please provide input video and output flow path. Aborting..." << endl;
        return -1;
    }
	cout << "Processing: " << argv[1] << endl;

    if (argc == 4){
        cout << "Using GPU #: " << argv[3]<< endl;
        setDevice(atoi(argv[3]));
    }
    else {
        cout << "Using GPU #: 0" << endl;
        setDevice(0);
    }

	// Parameters
	int bound = 20;

    // Capturing video
    VideoCapture cap(argv[1]);
    if(!cap.isOpened())
        return -1;

    Mat frame0, frame1;
    cap >> frame0;
    cv::cvtColor(frame0, frame0, CV_BGR2GRAY);
    cv::Size video_size = frame0.size();

    GpuMat d_flow, d_frame0, d_frame1, d_frame0f, d_frame1f;
    Ptr<cuda::OpticalFlowDual_TVL1> optFlow = cuda::OpticalFlowDual_TVL1::create();

    // Loss less compression. It should work with FFMpeg enabled.
    // cv::VideoWriter video_writer(argv[2], CV_FOURCC('F','F','V','1'), cap.get(CV_CAP_PROP_FPS), video_size);
    cv::VideoWriter video_writer(argv[2], CV_FOURCC('M','J','P','G'), cap.get(CV_CAP_PROP_FPS), video_size);
    while(true)
    {
        cap >> frame1;
        if (frame1.empty())
            break;

        cv::cvtColor(frame1, frame1, CV_BGR2GRAY);

        d_frame0.upload(frame0);
        d_frame1.upload(frame1);
        d_frame0.convertTo(d_frame0f, CV_32F, 1.0 / 255.0);
        d_frame1.convertTo(d_frame1f, CV_32F, 1.0 / 255.0);

        optFlow->calc(d_frame0f, d_frame1f, d_flow);
        GpuMat planes[2];
        cuda::split(d_flow, planes);

        Mat flow_x(planes[0]);
        Mat flow_y(planes[1]);
        std::vector<uchar> str_x, str_y, str_img;
        encodeFlowMap(flow_x, flow_y, str_x, str_y, bound, false);

        Mat single_img = createSingleImg(str_x, str_y, frame1.size());
        video_writer.write(single_img);

        std::swap(frame0, frame1);
    }
    cap.release();

    // Loss less compression. It should work with FFMpeg enabled.
    // cv::VideoWriter video_writer("abc.avi", CV_FOURCC('M','J','P','G'), cap.get(CV_CAP_PROP_FPS), video_size);
    // for(unsigned int i = 0; i < flow_frames.size(); i++){
    //     cout << flow_frames[i] << endl;
    //     video_writer.write(flow_frames[i]);
    // }
    video_writer.release();
    cout << "Flow saved at " << argv[2] << endl;

	return 0;
}