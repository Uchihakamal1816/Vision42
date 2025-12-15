import { useDropzone } from 'react-dropzone';
import { Upload, Image as ImageIcon } from 'lucide-react';
import { motion } from 'framer-motion';

export default function ImageDropZone({ onImageUpload, image }) {
  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    accept: {
      'image/*': ['.png', '.jpg', '.jpeg', '.tiff']
    },
    maxFiles: 1,
    onDrop: (acceptedFiles) => {
      if (acceptedFiles.length > 0) {
        onImageUpload(acceptedFiles[0]);
      }
    }
  });

  return (
    <div className="h-full flex flex-col">
      {image ? (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="flex-1 relative rounded-lg overflow-hidden border border-gray-300 dark:border-satellite-border"
        >
          <img
            src={image}
            alt="Satellite"
            className="w-full h-full object-contain bg-gray-100 dark:bg-black"
          />
        </motion.div>
      ) : (
        <div className="flex-1 flex items-center justify-center">
          <motion.div
            {...getRootProps()}
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.98 }}
            className={`w-full h-full border-2 border-dashed rounded-lg flex flex-col items-center justify-center cursor-pointer transition-all ${
              isDragActive
                ? 'border-blue-500 dark:border-space-500 bg-blue-50 dark:bg-space-900/30 shadow-lg dark:shadow-glow'
                : 'border-gray-300 dark:border-satellite-border hover:border-blue-400 dark:hover:border-space-600 hover:bg-gray-50 dark:hover:bg-satellite-card'
            }`}
          >
            <input {...getInputProps()} />
            <Upload className={`h-16 w-16 mb-4 ${isDragActive ? 'text-blue-500 dark:text-space-400' : 'text-gray-400 dark:text-gray-500'}`} />
            <p className="text-lg font-medium mb-2 text-gray-900 dark:text-white">
              {isDragActive ? 'Drop image here' : 'Drop a satellite image here'}
            </p>
            <button className="px-6 py-2 bg-blue-600 dark:bg-space-600 hover:bg-blue-700 dark:hover:bg-space-700 text-white rounded-lg transition-colors font-medium">
              Choose File
            </button>
          </motion.div>
        </div>
      )}
    </div>
  );
}
