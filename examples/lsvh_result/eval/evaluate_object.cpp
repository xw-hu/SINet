#include <iostream>
#include <algorithm>
#include <stdio.h>
#include <math.h>
#include <vector>
#include <numeric>
#include <strings.h>
#include <assert.h>
#include <fstream>
#include <iostream>

using namespace std;

/*=======================================================================
STATIC EVALUATION PARAMETERS
=======================================================================*/

// holds the number of test images on the server
int32_t N_VALIMAGES = 4310;



// sunny, rainy, blurry and occuluded evaluation level
enum SCENE{SUNNY=0, RAINY=1, BLURRY=2, OCCULUDED=3};

// evaluation parameter
const int32_t MIN_HEIGHT[4] = {15, 15, 15, 15};     // minimum height for evaluated groundtruth/detections
const int32_t MAX_OCCLUSION[4]  = {0, 1, 2, 3};        // maximum occlusion level of the groundtruth used for evaluation
const double  MAX_TRUNCATION[4] = {0.15, 0.3, 0.5, 0.5}; // maximum truncation level of the groundtruth used for evaluation


// evaluated object classes
enum CLASSES{CAR=0, BUS=1, VAN=2};

// parameters varying per class
vector<string> CLASS_NAMES;
const double   MIN_OVERLAP[4] = {0.7, 0.7, 0.7, 0.7};           // the minimum overlap required for evaluation

// no. of recall steps that should be evaluated (discretized)
const double N_SAMPLE_PTS = 41;

// initialize class names
void initGlobals () {
  CLASS_NAMES.push_back("1");
  CLASS_NAMES.push_back("2");
  CLASS_NAMES.push_back("3");
}

/*=======================================================================
DATA TYPES FOR EVALUATION
=======================================================================*/

// holding data needed for precision-recall and precision-aos
struct tPrData {
  vector<double> v;           // detection score for computing score thresholds
  double         similarity;  // orientation similarity
  int32_t        tp;          // true positives
  int32_t        fp;          // false positives
  int32_t        fn;          // false negatives
  tPrData () :
    similarity(0), tp(0), fp(0), fn(0) {}
};

// holding bounding boxes for ground truth and detections
struct tBox {
  string  type;     // object type as car, bus or van,...
  double   x1;      // left corner
  double   y1;      // top corner
  double   x2;      // right corner
  double   y2;      // bottom corner
  double   alpha;   // image orientation
  tBox (string type, double x1,double y1,double x2,double y2,double alpha) :
    type(type),x1(x1),y1(y1),x2(x2),y2(y2),alpha(alpha) {}
};

// holding ground truth data
struct tGroundtruth {
  tBox    box;        // object type, box, orientation
  double  truncation; // truncation 0..1
  int32_t occlusion;  // occlusion 0,1,2 (non, partly, fully)
  int  scene;  // Scene of image
  tGroundtruth () :
    box(tBox("invalild",-1,-1,-1,-1,-10)),truncation(-1),occlusion(-1),scene(-1) {}
  tGroundtruth (tBox box,double truncation,int32_t occlusion,int scene) :
	  box(box), truncation(truncation), occlusion(occlusion),scene(scene) {}
  tGroundtruth (string type,double x1,double y1,double x2,double y2,double alpha,double truncation,int32_t occlusion,int scene) :
	  box(tBox(type, x1, y1, x2, y2, alpha)), truncation(truncation), occlusion(occlusion), scene(scene) {}
};

// holding detection data
struct tDetection {
  tBox    box;    // object type, box, orientation
  double  thresh; // detection score
  int     scene;
  tDetection ():
    box(tBox("invalid",-1,-1,-1,-1,-10)),thresh(-1000),scene(-1) {}
  tDetection (tBox box,double thresh,int scene) :
	box(box), thresh(thresh), scene(scene) {}
  tDetection (string type,double x1,double y1,double x2,double y2,double alpha,double thresh,int scene) :
	box(tBox(type, x1, y1, x2, y2, alpha)), thresh(thresh), scene(scene) {}
};

/*=======================================================================
FUNCTIONS TO LOAD DETECTION AND GROUND TRUTH DATA ONCE, SAVE RESULTS
=======================================================================*/

vector<tDetection> loadDetections(string file_name, bool &compute_aos, bool &eval_car, bool &eval_bus, bool &eval_van, bool &success, int scene) {

  // holds all detections (ignored detections are indicated by an index vector
  vector<tDetection> detections;
  FILE *fp = fopen(file_name.c_str(),"r");
  if (!fp) {
    success = false;
    return detections;
  }
  while (!feof(fp)) {
    tDetection d;
    double trash;
    char str[255];
    if (fscanf(fp, "%s %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf",
                   str, &trash,    &trash,    &d.box.alpha,
                   &d.box.x1,   &d.box.y1, &d.box.x2, &d.box.y2,
                   &trash,      &trash,    &trash,    &trash, 
                   &trash,      &trash,    &trash,    &d.thresh )==16) {
      d.box.type = str;
      d.scene = scene;
      detections.push_back(d);

      // orientation=-10 is invalid, AOS is not evaluated if at least one orientation is invalid
      if(d.box.alpha==-10)
        compute_aos = false;

      // a class is only evaluated if it is detected at least once
      if(!eval_car && !strcasecmp(d.box.type.c_str(), "1"))
        eval_car = true;
      if(!eval_bus && !strcasecmp(d.box.type.c_str(), "2"))
        eval_bus = true;
      if(!eval_van && !strcasecmp(d.box.type.c_str(), "3"))
        eval_van = true;
    }
  }
  fclose(fp);
  success = true;
  return detections;
}

vector<tGroundtruth> loadGroundtruth(string file_name,bool &success, int scene) {

  // holds all ground truth (ignored ground truth is indicated by an index vector
  vector<tGroundtruth> groundtruth;
  FILE *fp = fopen(file_name.c_str(),"r");
  if (!fp) {
    success = false;
    return groundtruth;
  }
  while (!feof(fp)) {
    tGroundtruth g;
    double trash;
    char str[255];
    if (fscanf(fp, "%s %lf %d %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf",
                   str, &g.truncation, &g.occlusion, &g.box.alpha,
                   &g.box.x1,   &g.box.y1,     &g.box.x2,    &g.box.y2,
                   &trash,      &trash,        &trash,       &trash, 
                   &trash,      &trash,        &trash )==15) {
      g.box.type = str;
      g.scene = scene;
      groundtruth.push_back(g);
    }
  }
  fclose(fp);
  success = true;
  return groundtruth;
}

void saveStats (const vector<double> &precision, const vector<double> &aos, FILE *fp_det, FILE *fp_ori) {

  // save precision to file
  if(precision.empty())
    return;
  for (int32_t i=0; i<precision.size(); i++)
    fprintf(fp_det,"%f ",precision[i]);
  fprintf(fp_det,"\n");

  // save orientation similarity, only if there were no invalid orientation entries in submission (alpha=-10)
  if(aos.empty())
    return;
  for (int32_t i=0; i<aos.size(); i++)
    fprintf(fp_ori,"%f ",aos[i]);
  fprintf(fp_ori,"\n");
}

/*=======================================================================
EVALUATION HELPER FUNCTIONS
=======================================================================*/

// criterion defines whether the overlap is computed with respect to both areas (ground truth and detection)
// or with respect to box a or b (detection and "dontcare" areas)
inline double boxoverlap(tBox a, tBox b, int32_t criterion=-1){

  // overlap is invalid in the beginning
  double o = -1;

  // get overlapping area
  double x1 = max(a.x1, b.x1);
  double y1 = max(a.y1, b.y1);
  double x2 = min(a.x2, b.x2);
  double y2 = min(a.y2, b.y2);

  // compute width and height of overlapping area
  double w = x2-x1;
  double h = y2-y1;

  // set invalid entries to 0 overlap
  if(w<=0 || h<=0)
    return 0;

  // get overlapping areas
  double inter = w*h;
  double a_area = (a.x2-a.x1) * (a.y2-a.y1);
  double b_area = (b.x2-b.x1) * (b.y2-b.y1);

  // intersection over union overlap depending on users choice
  if(criterion==-1)     // union
    o = inter / (a_area+b_area-inter);
  else if(criterion==0) // bbox_a
    o = inter / a_area;
  else if(criterion==1) // bbox_b
    o = inter / b_area;

  // overlap
  return o;
}

vector<double> getThresholds(vector<double> &v, double n_groundtruth){

  // holds scores needed to compute N_SAMPLE_PTS recall values
  vector<double> t;

  // sort scores in descending order
  // (highest score is assumed to give best/most confident detections)
  sort(v.begin(), v.end(), greater<double>());

  // get scores for linearly spaced recall
  double current_recall = 0;
  for(int32_t i=0; i<v.size(); i++){

    // check if right-hand-side recall with respect to current recall is close than left-hand-side one
    // in this case, skip the current detection score
    double l_recall, r_recall, recall;
    l_recall = (double)(i+1)/n_groundtruth;
    if(i<(v.size()-1))
      r_recall = (double)(i+2)/n_groundtruth;
    else
      r_recall = l_recall;

    if( (r_recall-current_recall) < (current_recall-l_recall) && i<(v.size()-1))
      continue;

    // left recall is the best approximation, so use this and goto next recall step for approximation
    recall = l_recall;

    // the next recall step was reached
    t.push_back(v[i]);
    current_recall += 1.0/(N_SAMPLE_PTS-1.0);
  }
  return t;
}

void cleanData(CLASSES current_class, const vector<tGroundtruth> &gt, const vector<tDetection> &det, vector<int32_t> &ignored_gt, vector<tGroundtruth> &dc, vector<int32_t> &ignored_det, int32_t &n_gt, SCENE scene){

  // extract ground truth bounding boxes for current evaluation class
  for(int32_t i=0;i<gt.size(); i++){

    // only bounding boxes with a minimum height are used for evaluation
    double height = gt[i].box.y2 - gt[i].box.y1;

    // neighboring classes are ignored ("van" for "car" and "person_sitting" for "bus")
    // (lower/upper cases are ignored)
    int32_t valid_class;

    // all classes without a neighboring class
    if(!strcasecmp(gt[i].box.type.c_str(), CLASS_NAMES[current_class].c_str()))
      valid_class = 1;

    // classes with a neighboring class
    else if(!strcasecmp(CLASS_NAMES[current_class].c_str(), "3") && !strcasecmp("1", gt[i].box.type.c_str()))
      valid_class = 0;
    else if(!strcasecmp(CLASS_NAMES[current_class].c_str(), "3") && !strcasecmp("2", gt[i].box.type.c_str()))
      valid_class = 0;

    else if(!strcasecmp(CLASS_NAMES[current_class].c_str(), "2") && !strcasecmp("1", gt[i].box.type.c_str()))
      valid_class = 0;
    else if(!strcasecmp(CLASS_NAMES[current_class].c_str(), "2") && !strcasecmp("3", gt[i].box.type.c_str()))
      valid_class = 0;

    else if(!strcasecmp(CLASS_NAMES[current_class].c_str(), "1") && !strcasecmp("2", gt[i].box.type.c_str()))
      valid_class = 0;
    else if(!strcasecmp(CLASS_NAMES[current_class].c_str(), "1") && !strcasecmp("3", gt[i].box.type.c_str()))
      valid_class = 0;
    // classes not used for evaluation
    else
      valid_class = -1;

    // ground truth is ignored, if occlusion, truncation exceeds the scene or ground truth is too small
    // (doesn't count as FN nor TP, although detections may be assigned)
    bool ignore = false;
    if(height<MIN_HEIGHT[scene] || gt[i].scene!=scene)
    	ignore = true;



    // set ignored vector for ground truth
    // current class and not ignored (total no. of ground truth is detected for recall denominator)
    if(valid_class==1 && !ignore){
      ignored_gt.push_back(0);
      n_gt++;
    }

    // neighboring class, or current class but ignored
    else if(valid_class==0 || (ignore && valid_class==1))
      ignored_gt.push_back(1);

    // all other classes which are FN in the evaluation
    else
      ignored_gt.push_back(-1);
  }

  // extract dontcare areas
  for(int32_t i=0;i<gt.size(); i++)
    if(!strcasecmp("4", gt[i].box.type.c_str()))
      dc.push_back(gt[i]);

  // extract detections bounding boxes of the current class
  for(int32_t i=0;i<det.size(); i++){

    // neighboring classes are not evaluated
    int32_t valid_class;
    if(!strcasecmp(det[i].box.type.c_str(), CLASS_NAMES[current_class].c_str()))
      valid_class = 1;
    else
      valid_class = -1;
    
    bool ig_det = false;
    if(det[i].scene!=scene)
        ig_det = true;
    // set ignored vector for detections
    if(valid_class==1 && ig_det==false)
      ignored_det.push_back(0);
    else
      ignored_det.push_back(-1);
  }
}

tPrData computeStatistics(int32_t imageId,CLASSES current_class, const vector<tGroundtruth> &gt, const vector<tDetection> &det, const vector<tGroundtruth> &dc, const vector<int32_t> &ignored_gt, const vector<int32_t>  &ignored_det, bool compute_fp, bool compute_aos=false, double thresh=0, bool debug=false){

  tPrData stat = tPrData();
  const double NO_DETECTION = -10000000;
  vector<double> delta;            // holds angular difference for TPs (needed for AOS evaluation)
  vector<bool> assigned_detection; // holds wether a detection was assigned to a valid or ignored ground truth
  assigned_detection.assign(det.size(), false);
  vector<bool> ignored_threshold;
  ignored_threshold.assign(det.size(), false); // holds detections with a threshold lower than thresh if FP are computed

  // detections with a low score are ignored for computing precision (needs FP)
  if(compute_fp)
    for(int32_t i=0; i<det.size(); i++)
      if(det[i].thresh<thresh)
    //if(det[i].thresh<0.9)
        ignored_threshold[i] = true;
   //cout<<thresh<<endl;
  // evaluate all ground truth boxes
  for(int32_t i=0; i<gt.size(); i++){

    // this ground truth is not of the current or a neighboring class and therefore ignored
    if(ignored_gt[i]==-1)
      continue;

    /*=======================================================================
    find candidates (overlap with ground truth > 0.5) (logical len(det))
    =======================================================================*/
    int32_t det_idx          = -1;
    double valid_detection = NO_DETECTION;
    double max_overlap     = 0;

    // search for a possible detection
    bool assigned_ignored_det = false;
    
      

    for(int32_t j=0; j<det.size(); j++){

      // detections not of the current class, already assigned or with a low threshold are ignored
      if(ignored_det[j]==-1)
        continue;
      if(assigned_detection[j])
        continue;
      if(ignored_threshold[j])
        continue;

      // find the maximum score for the candidates and get idx of respective detection
      double overlap = boxoverlap(det[j].box, gt[i].box);

      // for computing recall thresholds, the candidate with highest score is considered
      if(!compute_fp && overlap>MIN_OVERLAP[current_class] && det[j].thresh>valid_detection){
        det_idx         = j;
        valid_detection = det[j].thresh;
      }

      // for computing pr curve values, the candidate with the greatest overlap is considered
      // if the greatest overlap is an ignored detection (min_height), the overlapping detection is used
      else if(compute_fp && overlap>MIN_OVERLAP[current_class] && (overlap>max_overlap || assigned_ignored_det) && ignored_det[j]==0){
        max_overlap     = overlap;
        det_idx         = j;
        valid_detection = 1;
        assigned_ignored_det = false;
	//cout<<i<<":  "<<j<<"  "<<det[j].box.x1<<"  "<<det[j].box.y1<<endl;
      }
      else if(compute_fp && overlap>MIN_OVERLAP[current_class] && valid_detection==NO_DETECTION && ignored_det[j]==1){
        det_idx              = j;
        valid_detection      = 1;
        assigned_ignored_det = true;
      }
    }

    /*=======================================================================
    compute TP, FP and FN
    =======================================================================*/

    // nothing was assigned to this valid ground truth
    if(valid_detection==NO_DETECTION && ignored_gt[i]==0)
      stat.fn++;

    // only evaluate valid ground truth <=> detection assignments (considering scene level)
    else if(valid_detection!=NO_DETECTION && (ignored_gt[i]==1 || ignored_det[det_idx]==1))
      assigned_detection[det_idx] = true;

    // found a valid true positive
    else if(valid_detection!=NO_DETECTION){

      // write highest score to threshold vector
      stat.tp++;
      stat.v.push_back(det[det_idx].thresh);
       

      // compute angular difference of detection and ground truth if valid detection orientation was provided
      if(compute_aos)
        delta.push_back(gt[i].box.alpha - det[det_idx].box.alpha);

      // clean up
      assigned_detection[det_idx] = true;
    }
     
  }
  // if FP are requested, consider stuff area
  if(compute_fp){
    
    // count fp
    for(int32_t i=0; i<det.size(); i++){	
      // count false positives if required (height smaller than required is ignored (ignored_det==1)
      if(!(assigned_detection[i] || ignored_det[i]==-1 || ignored_det[i]==1 || ignored_threshold[i]))
      {
        stat.fp++;
      }
    }
    
    // do not consider detections overlapping with stuff area
    int32_t nstuff = 0;

    for(int32_t i=0; i<dc.size(); i++){
      for(int32_t j=0; j<det.size(); j++){

        // detections not of the current class, already assigned, with a low threshold or a low minimum height are ignored
        if(assigned_detection[j])
          continue;
        if(ignored_det[j]==-1 || ignored_det[j]==1)
          continue;
        if(ignored_threshold[j])
          continue;

        // compute overlap and assign to stuff area, if overlap exceeds class specific value
        double overlap = boxoverlap(det[j].box, dc[i].box, 0);
        if(overlap>MIN_OVERLAP[current_class]){
          assigned_detection[j] = true;	  
          nstuff++; 
        }
      }
    }
    // FP = no. of all not to ground truth assigned detections - detections assigned to stuff areas
    stat.fp -= nstuff;

    // if all orientation values are valid, the AOS is computed
    if(compute_aos){
      vector<double> tmp;

      // FP have a similarity of 0, for all TP compute AOS
      tmp.assign(stat.fp, 0);
      for(int32_t i=0; i<delta.size(); i++)
        tmp.push_back((1.0+cos(delta[i]))/2.0);

      // be sure, that all orientation deltas are computed
      assert(tmp.size()==stat.fp+stat.tp);
      assert(delta.size()==stat.tp);

      // get the mean orientation similarity for this image
      if(stat.tp>0 || stat.fp>0)
        stat.similarity = accumulate(tmp.begin(), tmp.end(), 0.0);

      // there was neither a FP nor a TP, so the similarity is ignored in the evaluation
      else
        stat.similarity = -1;
    }
  }
  return stat;
}

/*=======================================================================
EVALUATE CLASS-WISE
=======================================================================*/

bool eval_class (FILE *fp_det, FILE *fp_ori, CLASSES current_class,const vector< vector<tGroundtruth> > &groundtruth,const vector< vector<tDetection> > &detections, bool compute_aos, vector<double> &precision, vector<double> &aos, SCENE scene) {

  // init
  int32_t n_gt=0;                                     // total no. of gt (denominator of recall)
  vector<double> v, thresholds;                       // detection scores, evaluated for recall discretization
  vector< vector<int32_t> > ignored_gt, ignored_det;  // index of ignored gt detection for current class/scene
  vector< vector<tGroundtruth> > dontcare;            // index of dontcare areas, included in ground truth

  // for all test images do
  for (int32_t i=0; i<N_VALIMAGES; i++){

    // holds ignored ground truth, ignored detections and dontcare areas for current frame
    vector<int32_t> i_gt, i_det;
    vector<tGroundtruth> dc;

    // only evaluate objects of current class and ignore occluded, truncated objects
    cleanData(current_class, groundtruth[i], detections[i], i_gt, dc, i_det, n_gt, scene);
    ignored_gt.push_back(i_gt);
    ignored_det.push_back(i_det);
    dontcare.push_back(dc);

    // compute statistics to get recall values
    tPrData pr_tmp = tPrData();
    pr_tmp = computeStatistics(i,current_class, groundtruth[i], detections[i], dc, i_gt, i_det, false);
    
    // add detection scores to vector over all images
    for(int32_t j=0; j<pr_tmp.v.size(); j++)
      v.push_back(pr_tmp.v[j]);
  }

  // get scores that must be evaluated for recall discretization
  thresholds = getThresholds(v, n_gt);
   
  // compute TP,FP,FN for relevant scores
  vector<tPrData> pr;
  pr.assign(thresholds.size(),tPrData());
  for (int32_t i=0; i<N_VALIMAGES; i++){

    //cout<<"thresholds.size: "<<thresholds.size()<<endl;
    // for all scores/recall thresholds do:
    for(int32_t t=0; t<thresholds.size(); t++){
      tPrData tmp = tPrData();
      tmp = computeStatistics(i,current_class, groundtruth[i], detections[i], dontcare[i],
                              ignored_gt[i], ignored_det[i], true, compute_aos, thresholds[t], t==30);
 
      // add no. of TP, FP, FN, AOS for current frame to total evaluation for current threshold
      pr[t].tp += tmp.tp;
      pr[t].fp += tmp.fp;
      pr[t].fn += tmp.fn;
      if(tmp.similarity!=-1)
        pr[t].similarity += tmp.similarity;
      
    }

  }
  
  // compute recall, precision and AOS
  vector<double> recall;
  precision.assign(N_SAMPLE_PTS, 0);
  if(compute_aos)
    aos.assign(N_SAMPLE_PTS, 0);
  double r=0;
  for (int32_t i=0; i<thresholds.size(); i++){
    r = pr[i].tp/(double)(pr[i].tp + pr[i].fn);
    recall.push_back(r);
    precision[i] = pr[i].tp/(double)(pr[i].tp + pr[i].fp);    
    if(compute_aos)
      aos[i] = pr[i].similarity/(double)(pr[i].tp + pr[i].fp);
  }

  // filter precision and AOS using max_{i..end}(precision)
  for (int32_t i=0; i<thresholds.size(); i++){
    precision[i] = *max_element(precision.begin()+i, precision.end());
    if(compute_aos)
      aos[i] = *max_element(aos.begin()+i, aos.end());
  }

  // save statisics and finish with success
  saveStats(precision, aos, fp_det, fp_ori);
	return true;
}

void saveAndPlotPlots(string dir_name,string file_name,string obj_type,vector<double> vals[],bool is_aos){

  char command[1024];

  // save plot data to file
  FILE *fp = fopen((dir_name + "/" + file_name + ".txt").c_str(),"w");
  for (int32_t i=0; i<(int)N_SAMPLE_PTS; i++)
    fprintf(fp,"%f %f %f %f %f\n",(double)i/(N_SAMPLE_PTS-1.0),vals[0][i],vals[1][i],vals[2][i],vals[3][i]);
  fclose(fp);
  
  // create png + eps
  for (int32_t j=0; j<2; j++) {

    // open file  
    FILE *fp = fopen((dir_name + "/" + file_name + ".gp").c_str(),"w");

    // save gnuplot instructions
    if (j==0) {
      fprintf(fp,"set term png size 450,315 font \"Helvetica\" 11\n");
      fprintf(fp,"set output \"%s.png\"\n",file_name.c_str());
    } else {
      fprintf(fp,"set term postscript eps enhanced color font \"Helvetica\" 20\n");
      fprintf(fp,"set output \"%s.eps\"\n",file_name.c_str());
    }
    
    // set labels and ranges
    fprintf(fp,"set size ratio 0.7\n");
    fprintf(fp,"set xrange [0:1]\n");
    fprintf(fp,"set yrange [0:1]\n");
    fprintf(fp,"set xlabel \"Recall\"\n");
    if (!is_aos) fprintf(fp,"set ylabel \"Precision\"\n");
    else         fprintf(fp,"set ylabel \"Orientation Similarity\"\n");
    obj_type[0] = toupper(obj_type[0]);
    fprintf(fp,"set title \"%s\"\n",obj_type.c_str());

    // line width
    int32_t   lw = 5;
    if (j==0) lw = 3;
   
    // plot error curve
    fprintf(fp,"plot ");
    fprintf(fp,"\"%s.txt\" using 1:2 title 'Sunny' with lines ls 1 lw %d,",file_name.c_str(),lw);
    fprintf(fp,"\"%s.txt\" using 1:3 title 'Rainy' with lines ls 2 lw %d,",file_name.c_str(),lw);
    fprintf(fp,"\"%s.txt\" using 1:4 title 'Blurry' with lines ls 3 lw %d",file_name.c_str(),lw);
    fprintf(fp,"\"%s.txt\" using 1:5 title 'Occuluded' with lines ls 4 lw %d",file_name.c_str(),lw);

    // close file
    fclose(fp);

    // run gnuplot => create png + eps
    sprintf(command,"cd %s; gnuplot %s",dir_name.c_str(),(file_name + ".gp").c_str());
    system(command);
  }

  // create pdf and crop
  sprintf(command,"cd %s; ps2pdf %s.eps %s_large.pdf",dir_name.c_str(),file_name.c_str(),file_name.c_str());
  system(command);
  sprintf(command,"cd %s; pdfcrop %s_large.pdf %s.pdf",dir_name.c_str(),file_name.c_str(),file_name.c_str());
  system(command);
  sprintf(command,"cd %s; rm %s_large.pdf",dir_name.c_str(),file_name.c_str());
  system(command);
}





bool eval(string gt_dir, string result_dir, string data_list_path){

  // set some global parameters
  initGlobals();

  string plot_dir = result_dir + "/plot";

  // create output directories
  system(("mkdir " + plot_dir).c_str());
  
  // load datalist
  vector<string> data_list;
  fstream fData;
  fData.open(data_list_path.c_str());
  string list_line;
  while (getline(fData,list_line,'\n')) {
    data_list.push_back(list_line);
  }
  fData.close();
  N_VALIMAGES = data_list.size();

  // hold detections and ground truth in memory
  vector< vector<tGroundtruth> > groundtruth;
  vector< vector<tDetection> >   detections;

  // holds wether orientation similarity shall be computed (might be set to false while loading detections)
  // and which labels where provided by this submission
  bool compute_aos=true, eval_car=false, eval_bus=false, eval_van=false;

  // for all images read groundtruth and detections
  printf("Loading detections...\n");
  for (int32_t i=0; i<N_VALIMAGES; i++) {

    // file name
    //char file_name[256];
    //sprintf(file_name,"%s.txt",data_list[i]);
    string file_name;
    file_name = data_list[i]+".txt";
    
    int ptr = 0;
    for(ptr;;ptr++)
        if(file_name[ptr]=='-')
            break;
    string scenetype = file_name.substr(0,ptr);
	int scene=-1;
	if (scenetype == "Sparse") scene = 0;
	if (scenetype == "Sparse") scene = 0;
	if (scenetype == "Crowded") scene = 3;
	if (scenetype == "Crowded") scene = 3;
        
    // read ground truth and result poses
    bool gt_success,det_success;
    vector<tGroundtruth> gt   = loadGroundtruth(gt_dir + "/" + file_name,gt_success,scene);
    vector<tDetection>   det  = loadDetections(result_dir + "/data/" + file_name, compute_aos, eval_car, eval_bus, eval_van,det_success,scene);
    groundtruth.push_back(gt);
    detections.push_back(det);

    // check for errors
    if (!gt_success) {
      printf("ERROR: Couldn't read: %s of ground truth. Please write me an email!\n", file_name.c_str());
      return false;
    }
    if (!det_success) {
      printf("ERROR: Couldn't read: %s\n", file_name.c_str());
      return false;
    }
  }
  printf("  done.\n");

  // holds pointers for result files
  FILE *fp_det=0, *fp_ori=0;

  // eval cars
  if(eval_car){
    fp_det = fopen((result_dir + "/stats_" + CLASS_NAMES[CAR] + "_detection.txt").c_str(),"w");
    if(compute_aos)
      fp_ori = fopen((result_dir + "/stats_" + CLASS_NAMES[CAR] + "_orientation.txt").c_str(),"w");
    vector<double> precision[4], aos[4];
    if(   !eval_class(fp_det,fp_ori,CAR,groundtruth,detections,compute_aos,precision[0],aos[0],SUNNY)
       || !eval_class(fp_det,fp_ori,CAR,groundtruth,detections,compute_aos,precision[1],aos[1],RAINY)
       || !eval_class(fp_det,fp_ori,CAR,groundtruth,detections,compute_aos,precision[2],aos[2],BLURRY)
       || !eval_class(fp_det,fp_ori,CAR,groundtruth,detections,compute_aos,precision[3],aos[3],OCCULUDED)){

      printf("Car evaluation failed.\n");
      return false;
    }
    fclose(fp_det);
    saveAndPlotPlots(plot_dir,CLASS_NAMES[CAR] + "_detection",CLASS_NAMES[CAR],precision,0);
    if(compute_aos){
      saveAndPlotPlots(plot_dir,CLASS_NAMES[CAR] + "_orientation",CLASS_NAMES[CAR],aos,1);
      fclose(fp_ori);
    }
  }

  // eval bus
  if(eval_bus){
    fp_det = fopen((result_dir + "/stats_" + CLASS_NAMES[BUS] + "_detection.txt").c_str(),"w");
    if(compute_aos)
      fp_ori = fopen((result_dir + "/stats_" + CLASS_NAMES[BUS] + "_orientation.txt").c_str(),"w");
    vector<double> precision[4], aos[4];
    if(   !eval_class(fp_det,fp_ori,BUS,groundtruth,detections,compute_aos,precision[0],aos[0],SUNNY)
       || !eval_class(fp_det,fp_ori,BUS,groundtruth,detections,compute_aos,precision[1],aos[1],RAINY)
       || !eval_class(fp_det,fp_ori,BUS,groundtruth,detections,compute_aos,precision[2],aos[2],BLURRY)
       || !eval_class(fp_det,fp_ori,BUS,groundtruth,detections,compute_aos,precision[3],aos[3],OCCULUDED)){
      printf("Pedestrian evaluation failed.\n");
      return false;
    }
    fclose(fp_det);
    saveAndPlotPlots(plot_dir,CLASS_NAMES[BUS] + "_detection",CLASS_NAMES[BUS],precision,0);
    if(compute_aos){
      fclose(fp_ori);
      saveAndPlotPlots(plot_dir,CLASS_NAMES[BUS] + "_orientation",CLASS_NAMES[BUS],aos,1);
    }
  }

  // eval vans
  if(eval_van){
    fp_det = fopen((result_dir + "/stats_" + CLASS_NAMES[VAN]  + "_detection.txt").c_str(),"w");
    if(compute_aos)
      fp_ori = fopen((result_dir + "/stats_" + CLASS_NAMES[VAN] + "_orientation.txt").c_str(),"w");
    vector<double> precision[4], aos[4];
    if(   !eval_class(fp_det,fp_ori,VAN,groundtruth,detections,compute_aos,precision[0],aos[0],SUNNY)
       || !eval_class(fp_det,fp_ori,VAN,groundtruth,detections,compute_aos,precision[1],aos[1],RAINY)
       || !eval_class(fp_det,fp_ori,VAN,groundtruth,detections,compute_aos,precision[2],aos[2],BLURRY)
       || !eval_class(fp_det,fp_ori,VAN,groundtruth,detections,compute_aos,precision[3],aos[3],OCCULUDED)){
      printf("Van evaluation failed.\n");
      return false;
    }
    fclose(fp_det);
    saveAndPlotPlots(plot_dir,CLASS_NAMES[VAN] + "_detection",CLASS_NAMES[VAN],precision,0);
    if(compute_aos){
      fclose(fp_ori);
      saveAndPlotPlots(plot_dir,CLASS_NAMES[VAN] + "_orientation",CLASS_NAMES[VAN],aos,1);
    }
  }

  // success
  return true;
}

int32_t main (int32_t argc,char *argv[]) {

  // we need 4 arguments!
  if (argc!=4) {
    cout << "Usage: ./eval_detection gt_dir result_dir list" << endl;
    return 1;
  }

  // read arguments
  string gt_dir = argv[1];
  string result_dir = argv[2];
  string data_list_path = argv[3];

  printf("Thank you for participating in our evaluation!\n");

  // run evaluation
  if (eval(gt_dir,result_dir,data_list_path)) {
    printf("The evaluation is done!\n");
  } else {
    system(("rm -r " + result_dir).c_str());
    printf("An error occured while processing your results.\n");
    printf("Please make sure that the data in your zip archive has the right format!\n");
  }

  return 0;
}

