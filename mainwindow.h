#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QWidget>
#include <QFile>
#include <QElapsedTimer>
#include <QGraphicsScene>
#include <QGraphicsPixmapItem>
#include <QCloseEvent>
#include <QFileDialog>
#include <QDialog>
#include <QPushButton>
#include <QLineEdit>
#include <QTextEdit>
#include <QLabel>
#include <QGridLayout>
#include <QDir>
#include <QStringList>
#include <QDirIterator>
#include <opencv2/opencv.hpp>


class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = 0);
    ~MainWindow();

private:
    QPushButton *graphBtn;
    QLineEdit *graphLineEdit;

    QPushButton *checkpointBtn;
    QLineEdit *checkpointLineEdit;

    QPushButton *imgBtn;
    QLineEdit *imgLineEdit;

    QPushButton *doInferBtn;
    QTextEdit *inferenceEdit;
    QLabel *_label;

    QGridLayout *myLayout;

private slots:
    void showGraphFile();
    void chooseImage();
    void showCheckpointDir();
    void doInference();
};

#endif // MAINWINDOW_H
