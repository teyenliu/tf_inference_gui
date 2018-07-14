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
#include <QGridLayout>


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

    QPushButton *doInferBtn;
    QTextEdit *inferenceEdit;

    QGridLayout *myLayout;

private slots:
    void showGraphFile();
    void showCheckpointDir();
    void doInference();
};

#endif // MAINWINDOW_H
