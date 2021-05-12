
#include <TensorFlowLite.h>

#include "tensorflow/lite/micro/kernels/micro_ops.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

#include "model_data.h"

// Globals used for compatibility with arduino style sketches.

namespace {
tflite::ErrorReporter* error_reporter = nullptr;
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output_type = nullptr;


// An area of Memory to use for input, output and intermediate Tensors.
constexpr int kTensorArenaSize = 100 * 1024;
static uint8_t tensor_arena[kTensorArenaSize];
}
void setup() {
//   // put your setup code here, to run once:

//   //Set up logging. Google style is to avoid globals or statics because of
//   //lifetime uncertainity.

//   static tflite::MicroErrorReporter micro_error_reporter;
//   error_reporter = &micro_error_reporter;

//   //Map the model into usable data structure.

//   model = tflite::GetModel(model_full_integer_tflite);
//   if (model->version() != TFLITE_SCHEMA_VERSION) {
//     TF_LITE_REPORT_ERROR(error_reporter,
//                          "Model provided is schema version %d not equal"
//                          "to supported version %d.",
//                          model->version(), TFLITE_SCHEMA_VERSION);
//     return;
//   }
//   //This pulls in all the operation implementations we need.
//   //static tflite::MicroMutableOpResolver<1> resolver;
//   static tflite::AllOpsResolver resolver;

//   //Build Interpreter to Run the Model.
//   static tflite::MicroInterpreter static_interpreter(model, resolver, tensor_arena,
//       kTensorArenaSize, error_reporter);
//       interpreter = &static_interpreter;
// // Allocate memory to Tensors.
// TfLiteStatus allocate_status = interpreter->AllocateTensors();
// if(allocate_status != kTfLiteOk){
//   TF_LITE_REPORT_ERROR(error_reporter,"AllocateTensors() Failed()");
 
// }
      

//   //Obtain pointer to the Model's Input.
//   input = interpreter->input(0);

}

void loop() {
  // put your main code here, to run repeatedly:
  Serial.println("Test Start");
    static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;

  //Map the model into usable data structure.

  model = tflite::GetModel(model_full_integer_tflite);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    TF_LITE_REPORT_ERROR(error_reporter,
                         "Model provided is schema version %d not equal"
                         "to supported version %d.",
                         model->version(), TFLITE_SCHEMA_VERSION);
    return;
  }
  delay(5000);
  //This pulls in all the operation implementations we need.
  //static tflite::MicroMutableOpResolver<1> resolver;
  static tflite::AllOpsResolver resolver;
 delay(5000);
  //Build Interpreter to Run the Model.
  static tflite::MicroInterpreter static_interpreter(model, resolver, tensor_arena,
      kTensorArenaSize, error_reporter);
      interpreter = &static_interpreter;
// Allocate memory to Tensors.
TfLiteStatus allocate_status = interpreter->AllocateTensors();
if(allocate_status != kTfLiteOk){
  TF_LITE_REPORT_ERROR(error_reporter,"AllocateTensors() Failed()");
}
 delay(5000);
}
