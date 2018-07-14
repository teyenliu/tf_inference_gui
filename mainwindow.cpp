#include <tensorflow/core/public/session.h>
#include <tensorflow/core/public/session_options.h>
#include <tensorflow/core/protobuf/meta_graph.pb.h>

#include "tensorflow/core/lib/core/status.h"

#include <iomanip>
#include <string>
#include <iostream>
#include "mainwindow.h"

typedef std::vector<std::pair<std::string, tensorflow::Tensor>> tensor_dict;


/**
 * @brief load a previous store model
 * @details [long description]
 *
 * in Python run:
 *
 *    saver = tf.train.Saver(tf.global_variables())
 *    saver.save(sess, './exported/my_model')
 *    tf.train.write_graph(sess.graph, '.', './exported/graph.pb, as_text=False)
 *
 * this relies on a graph which has an operation called `init` responsible to initialize all variables, eg.
 *
 *    sess.run(tf.global_variables_initializer())  # somewhere in the python file
 *
 * @param sess active tensorflow session
 * @param graph_fn path to graph file (eg. "./exported/graph.pb")
 * @param checkpoint_fn path to checkpoint file (eg. "./exported/my_model", optional)
 * @return status of reloading
 */
tensorflow::Status LoadModel(tensorflow::Session *sess, std::string graph_fn, std::string checkpoint_fn = "") {
    tensorflow::Status status;

    // Read in the protobuf graph we exported
    tensorflow::MetaGraphDef graph_def;
    status = ReadBinaryProto(tensorflow::Env::Default(), graph_fn, &graph_def);
    if (status != tensorflow::Status::OK())
        return status;

    // create the graph in the current session
    status = sess->Create(graph_def.graph_def());
    if (status != tensorflow::Status::OK())
        return status;

    // restore model from checkpoint, iff checkpoint is given
    if (checkpoint_fn != "") {

        const std::string restore_op_name = graph_def.saver_def().restore_op_name();
        const std::string filename_tensor_name = graph_def.saver_def().filename_tensor_name();

        tensorflow::Tensor filename_tensor(tensorflow::DT_STRING, tensorflow::TensorShape());
        filename_tensor.scalar<std::string>()() = checkpoint_fn;

        std::cout << "filename_tensor's scalar:" << filename_tensor.dims() << std::endl;

        tensor_dict feed_dict = {{filename_tensor_name, filename_tensor}};
        status = sess->Run(feed_dict,
        {},
        {restore_op_name},
                           nullptr);
        if (status != tensorflow::Status::OK())
            return status;
    } else {
        // virtual Status Run(const std::vector<std::pair<string, Tensor> >& inputs,
        //                  const std::vector<string>& output_tensor_names,
        //                  const std::vector<string>& target_node_names,
        //                  std::vector<Tensor>* outputs) = 0;
        status = sess->Run({}, {}, {"init"}, nullptr);
        if (status != tensorflow::Status::OK())
            return status;
    }

    return tensorflow::Status::OK();
}


MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent)
{
    setWindowTitle(tr("TensorFlow Inference"));

    graphBtn=new QPushButton;
    graphBtn->setText(tr("Select graph file:"));
    graphLineEdit=new QLineEdit;

    checkpointBtn=new QPushButton;
    checkpointBtn->setText(tr("Select checkpoint file:"));
    checkpointLineEdit=new QLineEdit;

    doInferBtn = new QPushButton;
    doInferBtn->setText(tr("Do inference example"));
    inferenceEdit = new QTextEdit;

    //layout
    myLayout=new QGridLayout(this);

    myLayout->addWidget(graphBtn,0,0);
    myLayout->addWidget(graphLineEdit,0,1);
    myLayout->addWidget(checkpointBtn,1,0);
    myLayout->addWidget(checkpointLineEdit,1,1);
    myLayout->addWidget(doInferBtn,2,0);
    myLayout->addWidget(inferenceEdit,2,1);

    QHBoxLayout *mainLayout =new QHBoxLayout();
    mainLayout->addLayout(myLayout);
    QWidget *placeholderWidget = new QWidget;
    placeholderWidget->setLayout(mainLayout);
    setCentralWidget(placeholderWidget);

    //signal and slot
    connect(graphBtn,SIGNAL(clicked()),this,SLOT(showGraphFile()));
    connect(checkpointBtn,SIGNAL(clicked()),this,SLOT(showCheckpointDir()));
    connect(doInferBtn,SIGNAL(clicked()),this,SLOT(doInference()));

}

MainWindow::~MainWindow()
{
}

void MainWindow::showGraphFile()
{
    QString fileName = QFileDialog::getOpenFileName(this,
                                             "open graph file dialog",
                                             "/home/",
                                             "Graph file(*.meta)");
    graphLineEdit->setText(fileName);
    QString croped_fileName=fileName.section(".",0,0);
    checkpointLineEdit->setText(croped_fileName);
}

void MainWindow::showCheckpointDir()
{
    /*
    QString fileName = QFileDialog::getExistingDirectory(this,
                                                  "open checkpoint dialog",
                                                  "/home/",
                                                  QFileDialog::ShowDirsOnly| QFileDialog::DontResolveSymlinks);
    */
}

void MainWindow::doInference()
{
    //const std::string graph_fn = "/home/liudanny/git/tensorflow_inference/inference/exported/my_model.meta";
    //const std::string checkpoint_fn = "/home/liudanny/git/tensorflow_inference/inference/exported/my_model";

    // prepare session
    tensorflow::Session *sess;
    tensorflow::SessionOptions options;
    TF_CHECK_OK(tensorflow::NewSession(options, &sess));
    TF_CHECK_OK(LoadModel(sess, graphLineEdit->text().toUtf8().constData(),
                          checkpointLineEdit->text().toUtf8().constData()));

    // prepare inputs
    tensorflow::TensorShape data_shape({1, 2});
    tensorflow::Tensor data(tensorflow::DT_FLOAT, data_shape);

    // same as in python file
    auto data_ = data.flat<float>().data();
    for (int i = 0; i < 2; ++i)
        data_[i] = 1;

    tensor_dict feed_dict = {
        { "input", data },
    };

    std::vector<tensorflow::Tensor> outputs;
    TF_CHECK_OK(sess->Run(feed_dict, {"output", "dense/kernel:0", "dense/bias:0"}, {}, &outputs));
    inferenceEdit->clear();
    inferenceEdit->append(QString::fromStdString(data.DebugString()));
    inferenceEdit->append(QString::fromStdString(outputs[0].DebugString()));
    inferenceEdit->append(QString::fromStdString(outputs[1].DebugString()));
    inferenceEdit->append(QString::fromStdString(outputs[2].DebugString()));

    //std::cout << "input           " << data.DebugString() << std::endl;
    //std::cout << "output          " << outputs[0].DebugString() << std::endl;
    //std::cout << "dense/kernel:0  " << outputs[1].DebugString() << std::endl;
    //std::cout << "dense/bias:0    " << outputs[2].DebugString() << std::endl;
}
