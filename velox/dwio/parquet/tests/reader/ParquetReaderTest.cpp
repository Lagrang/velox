/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "velox/dwio/parquet/reader/ParquetReader.h"
#include <arrow/builder.h>
#include <arrow/c/abi.h>
#include <arrow/c/bridge.h>
#include <arrow/record_batch.h>
#include <arrow/table_builder.h>
#include <arrow/type.h>
#include <arrow/type_fwd.h>
#include <arrow/util/decimal.h>
#include <arrow/util/macros.h>
#include <arrow/util/thread_pool.h>
#include <gtest/gtest.h>
#include <type/StringView.h>
#include <type/Type.h>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <string>
#include "velox/dwio/parquet/reader/ParquetColumnReader.h"
#include "velox/dwio/parquet/tests/ParquetReaderTestBase.h"
#include "velox/vector/arrow/Bridge.h"

using namespace facebook::velox;
using namespace facebook::velox::common;
using namespace facebook::velox::dwio::common;
using namespace facebook::velox::dwio::parquet;
using namespace facebook::velox::parquet;

namespace {
auto defaultPool = memory::getDefaultMemoryPool();
}

class ParquetReaderTest : public ParquetReaderTestBase {};

ParquetReader createReader(const std::string& path, const ReaderOptions& opts) {
  return ParquetReader(
      std::make_unique<BufferedInput>(
          std::make_shared<LocalReadFile>(path), opts.getMemoryPool()),
      opts);
}

TEST_F(ParquetReaderTest, parseSample) {
  // sample.parquet holds two columns (a: BIGINT, b: DOUBLE) and
  // 20 rows (10 rows per group). Group offsets are 153 and 614.
  // Data is in plain uncompressed format:
  //   a: [1..20]
  //   b: [1.0..20.0]
  const std::string sample(getExampleFilePath("sample.parquet"));

  ReaderOptions readerOptions{defaultPool.get()};
  ParquetReader reader = createReader(sample, readerOptions);
  EXPECT_EQ(reader.numberOfRows(), 20ULL);

  auto type = reader.typeWithId();
  EXPECT_EQ(type->size(), 2ULL);
  auto col0 = type->childAt(0);
  EXPECT_EQ(col0->type->kind(), TypeKind::BIGINT);
  auto col1 = type->childAt(1);
  EXPECT_EQ(col1->type->kind(), TypeKind::DOUBLE);
  EXPECT_EQ(type->childByName("a"), col0);
  EXPECT_EQ(type->childByName("b"), col1);
}

TEST_F(ParquetReaderTest, parseEmpty) {
  // empty.parquet holds two columns (a: BIGINT, b: DOUBLE) and
  // 0 rows.
  const std::string empty(getExampleFilePath("empty.parquet"));

  ReaderOptions readerOptions{defaultPool.get()};
  ParquetReader reader = createReader(empty, readerOptions);
  EXPECT_EQ(reader.numberOfRows(), 0ULL);

  auto type = reader.typeWithId();
  EXPECT_EQ(type->size(), 2ULL);
  auto col0 = type->childAt(0);
  EXPECT_EQ(col0->type->kind(), TypeKind::BIGINT);
  auto col1 = type->childAt(1);
  EXPECT_EQ(col1->type->kind(), TypeKind::DOUBLE);
  EXPECT_EQ(type->childByName("a"), col0);
  EXPECT_EQ(type->childByName("b"), col1);
}

TEST_F(ParquetReaderTest, parseDate) {
  // date.parquet holds a single column (date: DATE) and
  // 25 rows.
  // Data is in plain uncompressed format:
  //   date: [1969-12-27 .. 1970-01-20]
  const std::string sample(getExampleFilePath("date.parquet"));

  ReaderOptions readerOptions{defaultPool.get()};
  ParquetReader reader = createReader(sample, readerOptions);

  EXPECT_EQ(reader.numberOfRows(), 25ULL);

  auto type = reader.typeWithId();
  EXPECT_EQ(type->size(), 1ULL);
  auto col0 = type->childAt(0);
  EXPECT_EQ(col0->type->kind(), TypeKind::DATE);
  EXPECT_EQ(type->childByName("date"), col0);
}

TEST_F(ParquetReaderTest, parseRowMapArray) {
  // sample.parquet holds one row of type (ROW(BIGINT c0, MAP(VARCHAR,
  // ARRAY(INTEGER)) c1) c)
  const std::string sample(getExampleFilePath("row_map_array.parquet"));

  ReaderOptions readerOptions{defaultPool.get()};
  ParquetReader reader = createReader(sample, readerOptions);

  EXPECT_EQ(reader.numberOfRows(), 1ULL);

  auto type = reader.typeWithId();
  EXPECT_EQ(type->size(), 1ULL);

  auto col0 = type->childAt(0);
  EXPECT_EQ(col0->type->kind(), TypeKind::ROW);
  EXPECT_EQ(type->childByName("c"), col0);

  auto col0_0 = col0->childAt(0);
  EXPECT_EQ(col0_0->type->kind(), TypeKind::BIGINT);
  EXPECT_EQ(col0->childByName("c0"), col0_0);

  auto col0_1 = col0->childAt(1);
  EXPECT_EQ(col0_1->type->kind(), TypeKind::MAP);
  EXPECT_EQ(col0->childByName("c1"), col0_1);

  auto col0_1_0 = col0_1->childAt(0);
  EXPECT_EQ(col0_1_0->type->kind(), TypeKind::VARCHAR);

  auto col0_1_1 = col0_1->childAt(1);
  EXPECT_EQ(col0_1_1->type->kind(), TypeKind::ARRAY);

  auto col0_1_1_0 = col0_1_1->childAt(0);
  EXPECT_EQ(col0_1_1_0->type->kind(), TypeKind::INTEGER);
}

TEST_F(ParquetReaderTest, projectNoColumns) {
  // This is the case for count(*).
  auto rowType = ROW({}, {});
  ReaderOptions readerOpts{defaultPool.get()};
  ParquetReader reader =
      createReader(getExampleFilePath("sample.parquet"), readerOpts);
  RowReaderOptions rowReaderOpts;
  rowReaderOpts.setScanSpec(makeScanSpec(rowType));
  auto rowReader = reader.createRowReader(rowReaderOpts);
  auto result = BaseVector::create(rowType, 1, pool_.get());
  constexpr int kBatchSize = 100;
  ASSERT_TRUE(rowReader->next(kBatchSize, result));
  EXPECT_EQ(result->size(), 10);
  ASSERT_TRUE(rowReader->next(kBatchSize, result));
  EXPECT_EQ(result->size(), 10);
  ASSERT_FALSE(rowReader->next(kBatchSize, result));
}

TEST_F(ParquetReaderTest, parseIntDecimal) {
  // decimal_dict.parquet two columns (a: DECIMAL(7,2), b: DECIMAL(14,2)) and
  // 6 rows.
  // The physical type of the decimal columns:
  //   a: int32
  //   b: int64
  // Data is in dictionary encoding:
  //   a: [11.11, 11.11, 22.22, 22.22, 33.33, 33.33]
  //   b: [11.11, 11.11, 22.22, 22.22, 33.33, 33.33]
  auto rowType = ROW({"a", "b"}, {DECIMAL(7, 2), DECIMAL(14, 2)});
  ReaderOptions readerOpts{defaultPool.get()};
  const std::string decimal_dict(getExampleFilePath("decimal_dict.parquet"));

  ParquetReader reader = createReader(decimal_dict, readerOpts);
  RowReaderOptions rowReaderOpts;
  rowReaderOpts.setScanSpec(makeScanSpec(rowType));
  auto rowReader = reader.createRowReader(rowReaderOpts);

  EXPECT_EQ(reader.numberOfRows(), 6ULL);

  auto type = reader.typeWithId();
  EXPECT_EQ(type->size(), 2ULL);
  auto col0 = type->childAt(0);
  auto col1 = type->childAt(1);
  EXPECT_EQ(col0->type->kind(), TypeKind::SHORT_DECIMAL);
  EXPECT_EQ(col1->type->kind(), TypeKind::SHORT_DECIMAL);

  int64_t expectValues[3] = {1111, 2222, 3333};
  auto result = BaseVector::create(rowType, 1, pool_.get());
  rowReader->next(6, result);
  EXPECT_EQ(result->size(), 6ULL);
  auto decimals = result->as<RowVector>();
  auto a =
      decimals->childAt(0)->asFlatVector<UnscaledShortDecimal>()->rawValues();
  auto b =
      decimals->childAt(1)->asFlatVector<UnscaledShortDecimal>()->rawValues();
  for (int i = 0; i < 3; i++) {
    int index = 2 * i;
    EXPECT_EQ(a[index].unscaledValue(), expectValues[i]);
    EXPECT_EQ(a[index + 1].unscaledValue(), expectValues[i]);
    EXPECT_EQ(b[index].unscaledValue(), expectValues[i]);
    EXPECT_EQ(b[index + 1].unscaledValue(), expectValues[i]);
  }
}

int64_t AccountArray(const std::shared_ptr<arrow::ArrayData>& array) {
  int64_t bytes = 0;
  for (const auto& buf : array->buffers) {
    if (buf) {
      // Use capacity() instead of size() of buffer to account
      // the memory consumed by ingester tool.
      bytes += buf->capacity();
    }
  }
  if (array->dictionary) {
    bytes += AccountArray(array->dictionary);
  }
  for (const auto& childArray : array->child_data) {
    bytes += AccountArray(childArray);
  }

  return bytes;
}

TEST_F(ParquetReaderTest, scan) {
  namespace fs = std::filesystem;

  auto threadsStr = std::getenv("THREADS");
  int32_t threads = threadsStr != nullptr ? std::atoi(threadsStr)
                                          : std::thread::hardware_concurrency();
  std::string dirStr(std::getenv("ETL_DIR"));
  EXPECT_FALSE(dirStr.empty());
  auto path = fs::path(dirStr);

  std::vector<fs::path> paths;
  for (const fs::directory_entry& dir_entry :
       fs::recursive_directory_iterator(path)) {
    if (dir_entry.is_regular_file() && dir_entry.path().has_extension() &&
        dir_entry.path().extension() == ".parquet") {
      paths.emplace_back(dir_entry.path());
    }
  }

  auto start = std::chrono::steady_clock::now();
  arrow::Status status;
  std::mutex mutex;
  std::shared_ptr<arrow::internal::ThreadPool> pool =
      *arrow::internal::ThreadPool::Make(threads);
  std::vector<arrow::Future<arrow::internal::Empty>> futures;
  std::atomic_int64_t rows = 0;
  std::atomic_int64_t batches = 0;
  std::atomic_int64_t dataSize = 0;
  for (int i = 0; i < threads; ++i) {
    arrow::Result<arrow::Future<arrow::internal::Empty>> fut =
        pool->Submit([this, &paths, &mutex, &rows, &batches, &dataSize]() {
          while (true) {
            std::unique_lock lock(mutex);
            if (paths.empty()) {
              return arrow::Status::OK();
            }
            std::filesystem::path filePath = *(paths.end() - 1);
            paths.pop_back();
            lock.unlock();

            ReaderOptions readerOpts{defaultPool.get()};
            ParquetReader reader = createReader(filePath, readerOpts);
            RowReaderOptions rowReaderOpts;
            rowReaderOpts.setScanSpec(makeScanSpec(reader.rowType()));
            auto rowReader = reader.createRowReader(rowReaderOpts);

            auto result = BaseVector::create(reader.rowType(), 1, pool_.get());
            while (rowReader->next(128 * 1024, result) > 0) {
              ArrowArray arrowArray;
              ArrowSchema schema;
              facebook::velox::exportToArrow(result, schema);
              facebook::velox::exportToArrow(result, arrowArray, pool_.get());

              batches++;
              auto recBatch = *arrow::ImportRecordBatch(&arrowArray, &schema);
              rows += recBatch->num_rows();
              for (const auto& col : recBatch->columns()) {
                if (col->data()) {
                  dataSize += AccountArray(col->data());
                }
              }
            }
          }
          return arrow::Status::OK();
        });
    futures.emplace_back(*fut);
  }

  arrow::Status res;
  for (const auto& f : futures) {
    f.Wait();
    status &= f.status();
  }
  auto end = std::chrono::steady_clock::now();
  EXPECT_TRUE(status.ok());
  EXPECT_EQ(rows, 0);
  EXPECT_EQ(batches, 0);
  EXPECT_EQ((dataSize * 1.0 / 1024 * 1024), 0);
}
