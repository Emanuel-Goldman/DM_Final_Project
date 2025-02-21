import { FormField, Select, SelectProps } from "@cloudscape-design/components";
import { Option } from "./types";
import { useState } from "react";

interface  AttributeSelectProps {
    label: string;
    selectedAttr: Option | null;
    columns: Option[];
    handleSelection: SelectProps['onChange'];
}

export const AttributeSelect: React.FC<AttributeSelectProps> = ({label, selectedAttr, columns, handleSelection}) => {
    return (
        <FormField label={label} stretch>
            <Select
                selectedOption={selectedAttr}
                options={columns}
                placeholder="Select attribute ..."
                onChange={handleSelection}
            />
        </FormField>
    );
}

export const useAttributeSelection = () => {
    const [firstAttr, setFirstAttr] = useState<Option | null>(null);
    const [secondAttr, setSecondAttr] = useState<Option | null>(null);

    const handleFirstAttrSelect: SelectProps['onChange'] = (e) => {
        setFirstAttr(e.detail.selectedOption);
    };

    const handleSecondAttrSelect: SelectProps['onChange'] = (e) => {
        setSecondAttr(e.detail.selectedOption);
    };
    
    return {
        firstAttr,
        handleFirstAttrSelect,
        secondAttr,
        handleSecondAttrSelect
    };
};