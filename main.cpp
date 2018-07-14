// 2018, Patrick Wieschollek <mail@patwie.com>

#include "mainwindow.h"
#include <QApplication>


int main(int argc, char *argv[]) {

  // Qt application
  QApplication a(argc, argv);
  MainWindow w;
  w.show();
  return a.exec();
}
