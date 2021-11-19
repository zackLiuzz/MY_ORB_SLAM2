/**
* This file is part of ORB-SLAM2.
*
* Copyright (C) 2014-2016 Raúl Mur-Artal <raulmur at unizar dot es> (University of Zaragoza)
* For more information see <https://github.com/raulmur/ORB_SLAM2>
*
* ORB-SLAM2 is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM2 is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with ORB-SLAM2. If not, see <http://www.gnu.org/licenses/>.
*/


#ifndef ORBMATCHER_H
#define ORBMATCHER_H

#include<vector>
#include<opencv2/core/core.hpp>
#include<opencv2/features2d/features2d.hpp>

#include"MapPoint.h"
#include"KeyFrame.h"
#include"Frame.h"


namespace ORB_SLAM2
{
//该类负责特征点与特征点之间，地图点与特征点之间通过投影关系、词袋模型或者Sim3位姿匹配
class ORBmatcher
{    
public:
	   /**
	    * 找到在 以x, y为中心,边长为2r的方形内且尺度在[minLevel, maxLevel]的特征点
	    * @param nnratio        匹配特征点时，确定时候最好匹配与次好匹配差距的阈值。其值越小，其匹配越精确
	    * @param checkOri       是否开启匹配点的方向分类
	    */
    ORBmatcher(float nnratio=0.6, bool checkOri=true);

    // Computes the Hamming distance between two ORB descriptors
    static int DescriptorDistance(const cv::Mat &a, const cv::Mat &b);

    // Search matches between Frame keypoints and projected MapPoints. Returns number of matches
    // Used to track the local map (Tracking)
    //将F里的特征值与vpMapPoints进行匹配，通过投影加速
    //返回通过此函数匹配成功的数量
    int SearchByProjection(Frame &F, const std::vector<MapPoint*> &vpMapPoints, const float th=3);

    // Project MapPoints tracked in last frame into the current frame and search matches.
    // Used to track from previous frame (Tracking)
    /**根据上一帧LastFrame的特征点以及所对应的mappoint信息，寻找当前帧的哪些特征点与哪些mappoint的匹配联系
        * 根据上一帧特征点对应的3D点投影的位置缩小特征点匹配范围
        * @param  CurrentFrame 当前帧
        * @param  LastFrame    上一帧
        * @param  th           控制特征搜索框的大小阈值
        * @param  bMono        是否为单目
        * @return              成功匹配的数量
        */

    int SearchByProjection(Frame &CurrentFrame, const Frame &LastFrame, const float th, const bool bMono);

    // Project MapPoints seen in KeyFrame into the Frame and search matches.
    // Used in relocalisation (Tracking)
    /**在Tracking里的relocalisation中使用
        * CurrentFrame中特征点已经匹配好一些mappoint在sAlreadyFound中，通过此函数将pKF悉数投影到CurrentFrame再就近搜索特征点进行匹配
        * 也就是说CurrentFrame想通过这个函数在pKF的mappoint集合中匹配上更多的mappoint点
        * @param CurrentFrame 当前帧
        * @param pKF
        * @param sAlreadyFound CurrentFrame已经匹配上的mappoint
        * @param th 控制特征搜索框的大小阈值
        * @param ORBdist pKF中的mappoint是否能和CurrentFrame匹配成功的描述子距离的阈值
        * @return 成功匹配的数量
        */
    int SearchByProjection(Frame &CurrentFrame, KeyFrame* pKF, const std::set<MapPoint*> &sAlreadyFound, const float th, const int ORBdist);

    // Project MapPoints using a Similarity Transformation and search matches.
    // Used in loop detection (Loop Closing)
     int SearchByProjection(KeyFrame* pKF, cv::Mat Scw, const std::vector<MapPoint*> &vpPoints, std::vector<MapPoint*> &vpMatched, int th);

    // Search matches between MapPoints in a KeyFrame and ORB in a Frame.
    // Brute force constrained to ORB that belong to the same vocabulary node (at a certain level)
    // Used in Relocalisation and Loop Detection
    int SearchByBoW(KeyFrame *pKF, Frame &F, std::vector<MapPoint*> &vpMapPointMatches);
    int SearchByBoW(KeyFrame *pKF1, KeyFrame* pKF2, std::vector<MapPoint*> &vpMatches12);

    // Matching for the Map Initialization (only used in the monocular case)
    int SearchForInitialization(Frame &F1, Frame &F2, std::vector<cv::Point2f> &vbPrevMatched, std::vector<int> &vnMatches12, int windowSize=10);

    // Matching to triangulate new MapPoints. Check Epipolar Constraint.
    int SearchForTriangulation(KeyFrame *pKF1, KeyFrame* pKF2, cv::Mat F12,
                               std::vector<pair<size_t, size_t> > &vMatchedPairs, const bool bOnlyStereo);

    // Search matches between MapPoints seen in KF1 and KF2 transforming by a Sim3 [s12*R12|t12]
    // In the stereo and RGB-D case, s12=1
    int SearchBySim3(KeyFrame* pKF1, KeyFrame* pKF2, std::vector<MapPoint *> &vpMatches12, const float &s12, const cv::Mat &R12, const cv::Mat &t12, const float th);

    // Project MapPoints into KeyFrame and search for duplicated MapPoints.
    int Fuse(KeyFrame* pKF, const vector<MapPoint *> &vpMapPoints, const float th=3.0);

    // Project MapPoints into KeyFrame using a given Sim3 and search for duplicated MapPoints.
    int Fuse(KeyFrame* pKF, cv::Mat Scw, const std::vector<MapPoint*> &vpPoints, float th, vector<MapPoint *> &vpReplacePoint);

public:

    static const int TH_LOW; //匹配特征点时，描述子距离的阈值。特征点间描述子小于此值才考虑匹配
    static const int TH_HIGH;
    static const int HISTO_LENGTH;//按照匹配特征点之间的角度分类匹配特征点的数量


protected:
    //判断kp1，与kp2在基础矩阵F12下是否复合对极约束
    bool CheckDistEpipolarLine(const cv::KeyPoint &kp1, const cv::KeyPoint &kp2, const cv::Mat &F12, const KeyFrame *pKF);
    //根据观测角的cos值确定搜索区域的半径
    float RadiusByViewingCos(const float &viewCos);
    //找出数组histo中，vector.size()数量最大的前三位。也就是角度范围最多的前三位。
    void ComputeThreeMaxima(std::vector<int>* histo, const int L, int &ind1, int &ind2, int &ind3);
    //匹配特征点时，确定时候最好匹配与次好匹配差距的阈值。其值越小，其匹配越精确
    float mfNNratio;
    //是否开启匹配点角度差与其他大多数匹配点角度差差异较大的匹配点
    bool mbCheckOrientation;
};

}// namespace ORB_SLAM

#endif // ORBMATCHER_H
